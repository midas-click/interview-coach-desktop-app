"""faster-whisper transcription engine running in a background thread.

Pulls ``AudioChunk`` objects from a queue, transcribes buffered audio
periodically, and emits completed segments via a callback.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from src.audio.capture import AudioChunk

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
# engine
# ---------------------------------------------------------------------------


class TranscriptionEngine:
    """Streaming speech-to-text using faster-whisper.

    Runs its own daemon thread so blocking GPU/CPU inference never
    stalls the Qt event loop.
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
        self._sample_rate = 16000

        self._buffer_duration = buffer_duration
        self._min_samples = int(self._sample_rate * buffer_duration)
        self._overlap_samples = int(self._sample_rate * overlap_duration)

        # mutable state (guarded by _lock or only accessed in worker thread)
        self._audio_buffer: deque[tuple[np.ndarray, float]] = deque()
        self._buffer_len = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._processed_offset = 0.0  # cumulative seconds transcribed

        # set by start()
        self._on_segment: SegmentCallback | None = None

    # ----------------------------------------------------------------
    # public API
    # ----------------------------------------------------------------

    def start(
        self,
        audio_queue: queue.Queue[AudioChunk],
        on_segment: SegmentCallback,
    ) -> None:
        """Begin consuming *audio_queue* in a background thread.

        *on_segment* will be called from that thread whenever a new
        utterance is finalised.  It is the caller's responsibility to
        marshal those calls onto the UI thread (e.g. via Qt signals).
        """
        if self._running:
            return

        self._running = True
        self._processed_offset = 0.0
        self._on_segment = on_segment
        self._thread = threading.Thread(
            target=self._worker,
            args=(audio_queue,),
            daemon=True,
            name="transcriber",
        )
        self._thread.start()

    def stop(self) -> list[TranscriptionSegment]:
        """Signal the worker to stop and return any remaining segments."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)
        return self._flush()

    # ----------------------------------------------------------------
    # worker loop
    # ----------------------------------------------------------------

    def _worker(self, audio_queue: queue.Queue[AudioChunk]) -> None:
        while self._running:
            try:
                chunk = audio_queue.get(timeout=0.5)
            except queue.Empty:
                # no new audio — transcribe whatever is buffered
                if self._buffer_len >= self._sample_rate * 1.0:
                    self._transcribe_buffer()
                continue

            with self._lock:
                self._audio_buffer.append((chunk.data, chunk.timestamp))
                self._buffer_len += len(chunk.data)

            if self._buffer_len >= self._min_samples:
                self._transcribe_buffer()

    # ----------------------------------------------------------------
    # transcription
    # ----------------------------------------------------------------

    def _transcribe_buffer(self) -> None:
        """Transcribe accumulated audio, emit segments, keep overlap."""
        with self._lock:
            if self._buffer_len == 0:
                return
            audio, base_timestamp = self._assemble_buffer()
            keep = min(self._overlap_samples, self._buffer_len)
            self._trim_buffer(keep)

        segments, _info = self._model.transcribe(
            audio,
            vad_filter=True,
            word_timestamps=True,
            language="en",
            beam_size=5,
            best_of=5,
        )

        for seg in segments:
            start = base_timestamp + seg.start
            end = base_timestamp + seg.end
            self._processed_offset = max(self._processed_offset, end)
            self._emit(
                TranscriptionSegment(
                    text=seg.text.strip(),
                    start=round(start, 2),
                    end=round(end, 2),
                    confidence=round(seg.avg_log_prob, 3),
                )
            )

    def _flush(self) -> list[TranscriptionSegment]:
        """Transcribe everything remaining in the buffer (called on stop)."""
        segments: list[TranscriptionSegment] = []

        with self._lock:
            if self._buffer_len == 0:
                return segments
            audio, base_timestamp = self._assemble_buffer()
            self._buffer_len = 0
            self._audio_buffer.clear()

        raw_segments, _info = self._model.transcribe(
            audio,
            vad_filter=True,
            word_timestamps=True,
            language="en",
            beam_size=5,
            best_of=5,
        )

        for seg in raw_segments:
            segments.append(
                TranscriptionSegment(
                    text=seg.text.strip(),
                    start=round(base_timestamp + seg.start, 2),
                    end=round(base_timestamp + seg.end, 2),
                    confidence=round(seg.avg_log_prob, 3),
                )
            )
        return segments

    # ----------------------------------------------------------------
    # buffer helpers
    # ----------------------------------------------------------------

    def _assemble_buffer(self) -> tuple[np.ndarray, float]:
        """Concatenate buffered chunks and return (audio, base_timestamp)."""
        parts: list[np.ndarray] = []
        base_ts = self._audio_buffer[0][1] if self._audio_buffer else 0.0
        for data, _ts in self._audio_buffer:
            parts.append(data)
        return np.concatenate(parts), base_ts

    def _trim_buffer(self, keep_samples: int) -> None:
        """Keep only the last *keep_samples* frames to overlap with future audio."""
        if keep_samples <= 0:
            self._audio_buffer.clear()
            self._buffer_len = 0
            return

        remaining = keep_samples
        while self._audio_buffer and remaining > 0:
            data, ts = self._audio_buffer.popleft()
            if len(data) <= remaining:
                remaining -= len(data)
            else:
                # partial keep
                self._audio_buffer.appendleft((data[-remaining:], ts))
                remaining = 0

        self._buffer_len = sum(len(d) for d, _ in self._audio_buffer)

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
