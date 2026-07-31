"""faster-whisper transcription engine running in a background thread.

Pulls ``AudioChunk`` objects from a queue, routes them by source into
separate buffers, transcribes each independently, and emits segments
with the correct speaker label.

Audio is expected at 16 kHz.  If a device doesn't support 16 kHz the
capture layer falls back to its native rate, and we resample here.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from src.audio.capture import AudioChunk
from src.logger.logger import get_logger

log = get_logger(__name__)

SAMPLE_RATE = 16000


def _resample(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample to 16 kHz when a device doesn't support it natively."""
    if src_rate == SAMPLE_RATE:
        return audio
    new_len = int(len(audio) * SAMPLE_RATE / src_rate)
    old_idx = np.arange(len(audio), dtype=np.float64)
    new_idx = np.linspace(0, len(audio) - 1, new_len, dtype=np.float64)
    return np.interp(new_idx, old_idx, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# output type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    """A transcribed speech segment ready for storage / UI."""

    text: str
    start: float       # seconds from meeting start
    end: float
    confidence: float
    speaker: str = "Unknown"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _buffer_key(source: str) -> str:
    return "mic" if source == "microphone" else "sys"


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------


class TranscriptionEngine:
    """Streaming speech-to-text using faster-whisper.

    Maintains two independent audio buffers (mic / system audio) so
    each transcribed segment can be tagged with the correct speaker.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
        buffer_duration: float = 3.0,
        overlap_duration: float = 0.5,
    ) -> None:
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)

        self._buffer_duration = buffer_duration
        self._overlap_duration = overlap_duration

        self._buffers: dict[str, deque[tuple[np.ndarray, float]]] = {
            "mic": deque(), "sys": deque(),
        }
        self._buffer_lens: dict[str, int] = {"mic": 0, "sys": 0}
        self._lock = threading.Lock()

        self._running = False
        self._thread: threading.Thread | None = None
        self._on_segment: SegmentCallback | None = None
        self._final_segments: list[TranscriptionSegment] = []

    # ----------------------------------------------------------------
    # public API
    # ----------------------------------------------------------------

    def start(
        self,
        audio_queue: queue.Queue[AudioChunk],
        on_segment: SegmentCallback,
    ) -> None:
        if self._running:
            return
        self._running = True
        self._on_segment = on_segment
        self._thread = threading.Thread(
            target=self._worker, args=(audio_queue,),
            daemon=True, name="transcriber",
        )
        self._thread.start()

    def stop(self) -> list[TranscriptionSegment]:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30.0)
        return self._final_segments

    # ----------------------------------------------------------------
    # worker loop
    # ----------------------------------------------------------------

    def _worker(self, audio_queue: queue.Queue[AudioChunk]) -> None:
        log.info("Transcription worker started")
        while self._running:
            try:
                chunk = audio_queue.get(timeout=0.5)
            except queue.Empty:
                self._drain_idle()
                continue

            data = _resample(chunk.data, chunk.sample_rate)
            key = _buffer_key(chunk.source)
            with self._lock:
                self._buffers[key].append((data, chunk.timestamp))
                self._buffer_lens[key] += len(data)

            self._drain_buffers()

        self._final_segments = self._flush()
        log.info("Transcription worker stopped")

    def _drain_idle(self) -> None:
        for key in ("mic", "sys"):
            if self._buffer_lens[key] >= SAMPLE_RATE * 1.0:
                self._transcribe_buffer(key)

    def _drain_buffers(self) -> None:
        min_samples = int(SAMPLE_RATE * self._buffer_duration)
        for key in ("mic", "sys"):
            if self._buffer_lens[key] >= min_samples:
                self._transcribe_buffer(key)

    # ----------------------------------------------------------------
    # transcription
    # ----------------------------------------------------------------

    def _transcribe_buffer(self, key: str) -> None:
        with self._lock:
            buf_len = self._buffer_lens[key]
            if buf_len == 0:
                return
            audio, base_timestamp = self._assemble_buffer(key)
            keep = min(int(SAMPLE_RATE * self._overlap_duration), buf_len)
            self._trim_buffer(key, keep)

        duration = len(audio) / SAMPLE_RATE
        log.debug("Transcribing %.1fs of %s audio", duration, key)
        segments, _info = self._model.transcribe(
            audio,
            vad_filter=True,
            vad_parameters={"threshold": 0.35},
            word_timestamps=True,
            language="en",
            beam_size=5,
            best_of=1,
            initial_prompt="Interview conversation about technology, software engineering, and professional experience.",
        )

        seg_count = 0
        for seg in segments:
            seg_count += 1
            self._emit(self._make_segment(seg, base_timestamp, key))
        if seg_count:
            log.info("Transcribed %d segment(s) from %s", seg_count, key)

    def _flush(self) -> list[TranscriptionSegment]:
        segments: list[TranscriptionSegment] = []
        for key in ("mic", "sys"):
            with self._lock:
                buf_len = self._buffer_lens[key]
                if buf_len == 0:
                    continue
                audio, base_timestamp = self._assemble_buffer(key)
                self._buffer_lens[key] = 0
                self._buffers[key].clear()

            raw_segments, _info = self._model.transcribe(
                audio,
                vad_filter=True,
                vad_parameters={"threshold": 0.35},
                word_timestamps=True,
                language="en",
                beam_size=5,
                best_of=1,
            )
            for seg in raw_segments:
                segments.append(self._make_segment(seg, base_timestamp, key))
        return segments

    @staticmethod
    def _make_segment(whisper_seg, base_timestamp: float, source_key: str) -> TranscriptionSegment:
        return TranscriptionSegment(
            text=whisper_seg.text.strip(),
            start=round(base_timestamp + whisper_seg.start, 2),
            end=round(base_timestamp + whisper_seg.end, 2),
            confidence=round(whisper_seg.avg_logprob, 3),
            speaker="Me" if source_key == "mic" else "Interviewer",
        )

    # ----------------------------------------------------------------
    # buffer helpers
    # ----------------------------------------------------------------

    def _assemble_buffer(self, key: str) -> tuple[np.ndarray, float]:
        buf = self._buffers[key]
        parts: list[np.ndarray] = []
        base_ts = buf[0][1] if buf else 0.0
        for data, _ts in buf:
            parts.append(data)
        return np.concatenate(parts), base_ts

    def _trim_buffer(self, key: str, keep_samples: int) -> None:
        buf = self._buffers[key]
        if keep_samples <= 0:
            buf.clear()
            self._buffer_lens[key] = 0
            return

        current_len = self._buffer_lens[key]
        to_remove = current_len - keep_samples
        while buf and to_remove > 0:
            data, ts = buf.popleft()
            if len(data) <= to_remove:
                to_remove -= len(data)
            else:
                buf.appendleft((data[to_remove:], ts + to_remove / SAMPLE_RATE))
                to_remove = 0

        self._buffer_lens[key] = sum(len(d) for d, _ in buf)

    # ----------------------------------------------------------------
    # emission
    # ----------------------------------------------------------------

    def _emit(self, segment: TranscriptionSegment) -> None:
        if self._on_segment is not None:
            self._on_segment(segment)


# ---------------------------------------------------------------------------
# types
# ---------------------------------------------------------------------------

from collections.abc import Callable

SegmentCallback = Callable[[TranscriptionSegment], None]
