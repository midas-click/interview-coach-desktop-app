"""faster-whisper transcription engine running in a background thread.

Pulls ``AudioChunk`` objects from a queue, routes them by source into
separate buffers, transcribes each independently in a dedicated
executor thread, and emits segments with the correct speaker label.

Audio is expected at 16 kHz.  When a device doesn't natively support
16 kHz the capture layer falls back to its native rate, and we
resample here.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from src.audio.capture import AudioChunk
from src.logger.logger import get_logger

log = get_logger(__name__)

SAMPLE_RATE = 16000


def _resample(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample to 16 kHz when a device doesn't support it natively.

    For integer-ratio downsampling (e.g. 48→16 kHz, factor 3) uses
    sample averaging which provides basic anti-aliasing.  For non-integer
    ratios falls back to linear interpolation with a warning.
    """
    if src_rate == SAMPLE_RATE:
        return audio

    if src_rate % SAMPLE_RATE == 0:
        # Integer-ratio decimation via averaging — acts as a crude
        # low-pass filter at fs / (2 * factor).
        factor = src_rate // SAMPLE_RATE
        trimmed_len = len(audio) - len(audio) % factor
        if trimmed_len == 0:
            return np.array([], dtype=np.float32)
        return audio[:trimmed_len].reshape(-1, factor).mean(axis=1).astype(np.float32)

    # Non-integer ratio — rare (e.g. 44100→16000).  Linear interpolation
    # is a poor resampler, but this path is only hit on unusual hardware.
    log.warning(
        "Non-integer resampling %d→%d Hz — transcription quality may suffer. "
        "Consider using a 16 kHz or 48 kHz device.",
        src_rate, SAMPLE_RATE,
    )
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

    Audio ingestion runs in a worker thread.  Transcription is offloaded
    to a single-thread executor so the worker never blocks on the model.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
        buffer_duration: float = 2.0,
        overlap_duration: float = 0.5,
    ) -> None:
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
        log.info("Whisper model loaded: %s on %s (%s)", model_name, device, compute_type)

        self._buffer_duration = buffer_duration
        self._overlap_duration = overlap_duration

        self._buffers: dict[str, deque[tuple[np.ndarray, float]]] = {
            "mic": deque(), "sys": deque(),
        }
        self._buffer_lens: dict[str, int] = {"mic": 0, "sys": 0}
        self._lock = threading.Lock()

        # Sample-count-based clock — avoids clock drift between the
        # system monotonic clock and the audio device hardware clock.
        self._sample_offsets: dict[str, float] = {"mic": 0.0, "sys": 0.0}

        # Prevent piling up concurrent transcription jobs for one source.
        self._jobs_in_flight: dict[str, bool] = {"mic": False, "sys": False}

        self._running = False
        self._thread: threading.Thread | None = None
        self._executor: ThreadPoolExecutor | None = None
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

        # Fresh state for each session — the executor dies on stop().
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._buffers = {"mic": deque(), "sys": deque()}
        self._buffer_lens = {"mic": 0, "sys": 0}
        self._sample_offsets = {"mic": 0.0, "sys": 0.0}
        self._jobs_in_flight = {"mic": False, "sys": False}

        self._thread = threading.Thread(
            target=self._worker, args=(audio_queue,),
            daemon=True, name="transcriber",
        )
        self._thread.start()

    def stop(self) -> list[TranscriptionSegment]:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30.0)
        # Wait for any in-flight transcription to finish.
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        # Flush whatever audio remains in the buffers.
        self._final_segments = self._flush()
        return self._final_segments

    # ----------------------------------------------------------------
    # worker loop (audio ingestion — never blocks on transcription)
    # ----------------------------------------------------------------

    def _worker(self, audio_queue: queue.Queue[AudioChunk]) -> None:
        log.info("Transcription worker started")
        while self._running:
            try:
                chunk = audio_queue.get(timeout=0.5)
            except queue.Empty:
                try:
                    self._drain_idle()
                except Exception:
                    log.exception("drain_idle crashed")
                continue

            try:
                data = _resample(chunk.data, chunk.sample_rate)
                key = _buffer_key(chunk.source)
                with self._lock:
                    # Use sample-count timestamps so timing stays locked
                    # to the audio hardware clock, not the system clock.
                    ts = self._sample_offsets[key]
                    self._buffers[key].append((data, ts))
                    self._buffer_lens[key] += len(data)
                    self._sample_offsets[key] += len(data) / SAMPLE_RATE
                self._drain_buffers()
            except Exception:
                log.exception("Worker failed processing audio chunk")

        log.info("Transcription worker stopped")

    # ----------------------------------------------------------------
    # drain helpers — never call model.transcribe() here
    # ----------------------------------------------------------------

    def _drain_buffers(self) -> None:
        min_samples = int(SAMPLE_RATE * self._buffer_duration)
        for key in ("mic", "sys"):
            self._maybe_submit(key, min_samples)

    def _drain_idle(self) -> None:
        """Called when no new chunks arrived for 0.5 s — flush partial buffers."""
        for key in ("mic", "sys"):
            self._maybe_submit(key, int(SAMPLE_RATE * 1.0))

    def _maybe_submit(self, key: str, min_samples: int) -> None:
        """If *key* has enough audio and no job is in flight, submit a
        transcription task to the executor."""
        with self._lock:
            if self._buffer_lens[key] < min_samples or self._jobs_in_flight[key]:
                return
            self._jobs_in_flight[key] = True
            audio, base_timestamp = self._assemble_buffer(key)
            keep = min(int(SAMPLE_RATE * self._overlap_duration), self._buffer_lens[key])
            self._trim_buffer(key, keep)

        future = self._executor.submit(
            self._transcribe_audio, audio, base_timestamp, key,
        )
        future.add_done_callback(
            lambda f, k=key: self._on_transcription_done(f, k),
        )

    # ----------------------------------------------------------------
    # transcription (runs in executor thread)
    # ----------------------------------------------------------------

    def _transcribe_audio(
        self, audio: np.ndarray, base_timestamp: float, key: str,
    ) -> list[TranscriptionSegment]:
        duration = len(audio) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        log.debug("Transcribing %.1fs of %s audio (RMS: %.4f)", duration, key, rms)

        raw_segments, _info = self._model.transcribe(
            audio,
            vad_filter=True,
            word_timestamps=True,
            language="en",
            beam_size=5,
            best_of=1,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
        )
        results: list[TranscriptionSegment] = []
        for seg in raw_segments:
            results.append(self._make_segment(seg, base_timestamp, key))
        return results

    def _on_transcription_done(self, future, key: str) -> None:
        """Callback invoked by the executor when transcription finishes."""
        try:
            segments = future.result()
        except Exception:
            log.exception("Transcription failed for %s audio", key)
            segments = []

        with self._lock:
            self._jobs_in_flight[key] = False

        seg_count = 0
        for seg in segments:
            seg_count += 1
            self._emit(seg)
        if seg_count:
            log.info("Transcribed %d segment(s) from %s", seg_count, key)
        else:
            log.debug("No speech detected in %s audio", key)

    def _flush(self) -> list[TranscriptionSegment]:
        """Transcribe all remaining buffered audio synchronously (called at stop)."""
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
                word_timestamps=True,
                language="en",
                beam_size=5,
                best_of=1,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
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

SegmentCallback = Callable[[TranscriptionSegment], None]
