"""faster-whisper transcription engine running in a background thread.

Pulls ``AudioChunk`` objects from a queue, routes them by source into
separate buffers, transcribes each independently, and emits segments
with the correct speaker label.
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


def _speaker(source: str) -> str:
    return "Me" if source == "microphone" else "Interviewer"


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
        self._sample_rate: int | None = None

        self._buffer_duration = buffer_duration
        self._overlap_duration = overlap_duration

        # per-source buffers: "mic" / "sys"
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

            key = _buffer_key(chunk.source)
            with self._lock:
                self._buffers[key].append((chunk.data, chunk.timestamp))
                self._buffer_lens[key] += len(chunk.data)
                if self._sample_rate is None and chunk.sample_rate > 0:
                    self._sample_rate = chunk.sample_rate

            self._drain_buffers()

        self._final_segments = self._flush()
        log.info("Transcription worker stopped")

    def _drain_idle(self) -> None:
        """Transcribe partial buffers when no new audio arrives."""
        sr = self._sample_rate or 16000
        for key in ("mic", "sys"):
            if self._buffer_lens[key] >= sr * 1.0:
                self._transcribe_buffer(key)

    def _drain_buffers(self) -> None:
        sr = self._sample_rate or 16000
        min_samples = int(sr * self._buffer_duration)
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
            sr = self._sample_rate or 16000
            keep = min(int(sr * self._overlap_duration), buf_len)
            self._trim_buffer(key, keep)

        duration = len(audio) / sr
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
                initial_prompt="Interview conversation about technology, software engineering, and professional experience.",
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

        sr = self._sample_rate or 16000
        current_len = self._buffer_lens[key]
        to_remove = current_len - keep_samples
        while buf and to_remove > 0:
            data, ts = buf.popleft()
            if len(data) <= to_remove:
                to_remove -= len(data)
            else:
                buf.appendleft((data[to_remove:], ts + to_remove / sr))
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
