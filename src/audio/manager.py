"""Capture orchestrator — drains the audio queue, runs VAD per source,
writes utterance WAV files, and inserts queued records into SQLite.

Runs in its own thread. Never touches Whisper. Never blocks audio capture.
"""

from __future__ import annotations

import queue
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import wave
from silero_vad import load_silero_vad

from src.audio.capture import AudioCapture, AudioChunk, LoopbackCapture
from src.audio.vad import SAMPLE_RATE, UtteranceData, VAD
from src.logger.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

log = get_logger(__name__)

WAV_SAMPLE_WIDTH = 2  # int16 = 2 bytes per sample


def _write_wav(path: Path, audio: np.ndarray) -> None:
    """Write float32 audio as 16-bit PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(WAV_SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(int16.tobytes())


class AudioManager:
    """Manages two capture sources feeding VAD-driven utterance detection.

    Callbacks
    ---------
    ``on_speaking_change(speaker: str | None)``
        Called from the capture thread when the active speaker changes.
        ``"interviewer"``, ``"candidate"``, or ``None`` (silence).
    """

    def __init__(
        self,
        db_path: str,
        temp_dir: Path,
        vad_threshold: float = 0.5,
        vad_silence_ms: int = 800,
        vad_min_speech_ms: int = 250,
    ) -> None:
        self._db_path = db_path
        self._temp_dir = temp_dir

        self._queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=256)
        self._mic: AudioCapture | None = None
        self._sys: LoopbackCapture | None = None

        self._vad_model = load_silero_vad()
        self._vad_threshold = vad_threshold
        self._vad_silence_ms = vad_silence_ms
        self._vad_min_speech_ms = vad_min_speech_ms

        self._running = False
        self._thread: threading.Thread | None = None
        self._interview_id: str | None = None

        # per-source VAD instances (created in start)
        self._vads: dict[str, VAD] = {}

        # speaking tracking for callbacks
        self._last_speaker: str | None = None

        # callback
        self.on_speaking_change: Callable[[str | None], None] | None = None

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    @property
    def audio_queue(self) -> queue.Queue[AudioChunk]:
        return self._queue

    def start(self, interview_id: str, mic_device: int | None = None, sys_device: int | None = None) -> None:
        if self._running:
            return

        self._interview_id = interview_id
        self._temp_dir.mkdir(parents=True, exist_ok=True)

        self._vads["candidate"] = VAD(
            self._vad_model,
            threshold=self._vad_threshold,
            silence_ms=self._vad_silence_ms,
            min_speech_ms=self._vad_min_speech_ms,
        )
        self._vads["interviewer"] = VAD(
            self._vad_model,
            threshold=self._vad_threshold,
            silence_ms=self._vad_silence_ms,
            min_speech_ms=self._vad_min_speech_ms,
        )

        # start audio capture sources
        self._mic = AudioCapture(source_label="microphone", device_index=mic_device)
        self._mic.start(self._queue)

        if sys_device is not None:
            try:
                self._sys = LoopbackCapture(device_index=sys_device)
                self._sys.start(self._queue)
            except Exception:
                log.warning("System audio device %d failed", sys_device)

        self._running = True
        self._last_speaker = None
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="capture")
        self._thread.start()
        log.info("Audio capture started — interview=%s", interview_id)

    def stop_all(self) -> None:
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)

        # stop capture sources
        if self._mic:
            self._mic.stop()
            self._mic = None
        if self._sys:
            self._sys.stop()
            self._sys = None

        # drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        self._notify_speaker(None)
        log.info("Audio capture stopped")

    # ------------------------------------------------------------------
    # capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Drain the audio queue, feed VAD, write WAVs, insert SQLite.

        Runs until stop_all() sets self._running = False.
        """
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        log.info("Capture DB connection opened: %s", self._db_path)

        chunk_count = 0
        try:
            while self._running:
                try:
                    chunk = self._queue.get(timeout=0.5)
                except queue.Empty:
                    self._update_speaking_status()
                    continue

                chunk_count += 1
                speaker = "candidate" if chunk.source == "microphone" else "interviewer"
                vad = self._vads.get(speaker)
                if vad is None:
                    continue

                rms = float(np.sqrt(np.mean(chunk.data.astype(np.float64) ** 2)))
                if chunk_count % 10 == 1:
                    log.info("Audio chunk #%d: %s, %d samples @ %d Hz, RMS=%.4f, queue≈%d",
                             chunk_count, speaker, len(chunk.data), chunk.sample_rate,
                             rms, self._queue.qsize())

                prev_state = vad.state
                result = vad.feed(chunk.data, chunk.sample_rate)
                if vad.state != prev_state:
                    log.info("VAD %s: %s → %s (queue≈%d)", speaker, prev_state, vad.state, self._queue.qsize())

                if result is not None:
                    self._handle_utterance(conn, result, speaker)

                self._update_speaking_status()

            # flush remaining audio on stop
            for speaker, vad in self._vads.items():
                result = vad.flush()
                if result is not None:
                    log.info("Flush — final utterance from %s: %d–%d ms",
                             speaker, result.start_ms, result.end_ms)
                    self._handle_utterance(conn, result, speaker)
            log.info("Capture loop finished — %d chunks processed", chunk_count)
        finally:
            # Flush WAL into main database so external tools can read it
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()
            log.info("Capture DB connection closed (WAL checkpointed)")

    def _handle_utterance(self, conn: sqlite3.Connection, data: UtteranceData, speaker: str) -> None:
        wav_path = (
            self._temp_dir / self._interview_id / f"{uuid.uuid4().hex}.wav"
        )
        _write_wav(wav_path, data.audio)

        conn.execute(
            "INSERT INTO utterances (interview_id, speaker, start_ms, end_ms, audio_path, status) "
            "VALUES (?, ?, ?, ?, ?, 'queued')",
            (self._interview_id, speaker, data.start_ms, data.end_ms, str(wav_path)),
        )
        conn.commit()
        log.info("Utterance queued → DB: %s %d–%d ms (%d samples, wav=%s)",
                 speaker, data.start_ms, data.end_ms, len(data.audio), wav_path.name)

    def _update_speaking_status(self) -> None:
        """Notify UI if the active speaker changed."""
        active: str | None = None
        if self._vads.get("interviewer") and self._vads["interviewer"].is_speaking:
            active = "interviewer"
        elif self._vads.get("candidate") and self._vads["candidate"].is_speaking:
            active = "candidate"

        if active != self._last_speaker:
            self._last_speaker = active
            self._notify_speaker(active)

    def _notify_speaker(self, speaker: str | None) -> None:
        if self.on_speaking_change:
            try:
                self.on_speaking_change(speaker)
            except Exception:
                log.exception("on_speaking_change callback failed")
