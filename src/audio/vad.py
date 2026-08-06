"""Endpoint detection using silero-vad.

Tracks speech state per audio source, accumulating audio until a silence
gap of sufficient duration is detected, then emits a complete utterance.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import torch

from src.logger.logger import get_logger

log = get_logger(__name__)

SAMPLE_RATE = 16000
FRAME_SIZE = 512


def _resample(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample to 16 kHz via integer decimation or linear interpolation."""
    if src_rate == SAMPLE_RATE:
        return audio

    if src_rate % SAMPLE_RATE == 0:
        factor = src_rate // SAMPLE_RATE
        trimmed_len = len(audio) - len(audio) % factor
        if trimmed_len == 0:
            return np.array([], dtype=np.float32)
        return audio[:trimmed_len].reshape(-1, factor).mean(axis=1).astype(np.float32)

    log.warning(
        "Non-integer resampling %d→%d Hz — VAD accuracy may suffer.",
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
class UtteranceData:
    audio: np.ndarray   # float32, 16kHz mono
    start_ms: int
    end_ms: int


# ---------------------------------------------------------------------------
# VAD state machine
# ---------------------------------------------------------------------------


class VAD:
    """Per-source endpoint detector backed by silero-vad."""

    def __init__(
        self,
        model,
        threshold: float = 0.5,
        silence_ms: int = 800,
        min_speech_ms: int = 250,
    ) -> None:
        self._model = model
        self._threshold = threshold
        self._silence_samples = int(SAMPLE_RATE * silence_ms / 1000)
        self._min_speech_samples = int(SAMPLE_RATE * min_speech_ms / 1000)
        self._lock = threading.Lock()

        self._state: str = "idle"  # idle | speaking | silence
        self._buffer: list[np.ndarray] = []
        self._buffer_samples: int = 0
        self._speech_samples: int = 0
        self._silence_counter: int = 0
        self._start_sample: int = 0
        self._sample_counter: int = 0

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    @property
    def is_speaking(self) -> bool:
        return self._state == "speaking"

    @property
    def state(self) -> str:
        return self._state

    def feed(self, audio: np.ndarray, src_rate: int) -> UtteranceData | None:
        """Feed audio samples. Returns an utterance when an endpoint is detected."""
        if src_rate != SAMPLE_RATE:
            audio = _resample(audio, src_rate)

        result: UtteranceData | None = None

        with self._lock:
            for i in range(0, len(audio), FRAME_SIZE):
                frame = audio[i:i + FRAME_SIZE]
                if len(frame) < FRAME_SIZE:
                    break

                speech_prob = float(self._model(torch.from_numpy(frame), SAMPLE_RATE).item())
                self._advance(frame, speech_prob)

            # Check if we emitted during _advance
            if self._emitted:
                result = self._emitted
                self._emitted = None

        return result

    def flush(self) -> UtteranceData | None:
        """Force-emit any pending speech buffer (called on capture stop)."""
        with self._lock:
            if self._state in ("speaking", "silence") and self._buffer_samples >= self._min_speech_samples:
                return self._emit()
            return None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _advance(self, frame: np.ndarray, speech_prob: float) -> None:
        if self._state == "idle":
            if speech_prob > self._threshold:
                self._state = "speaking"
                self._start_sample = self._sample_counter
                self._buffer = [frame]
                self._buffer_samples = len(frame)
            # else: discard frame, stay idle

        elif self._state == "speaking":
            self._buffer.append(frame)
            self._buffer_samples += len(frame)
            self._speech_samples += len(frame)
            if speech_prob < self._threshold:
                self._state = "silence"
                self._silence_counter = len(frame)
            else:
                self._silence_counter = 0

        elif self._state == "silence":
            self._buffer.append(frame)
            self._buffer_samples += len(frame)
            if speech_prob > self._threshold:
                self._state = "speaking"
                self._silence_counter = 0
            else:
                self._silence_counter += len(frame)
                if self._silence_counter >= self._silence_samples:
                    if self._speech_samples >= self._min_speech_samples:
                        self._emitted = self._emit()
                    self._reset()

        self._sample_counter += len(frame)

    def _emit(self) -> UtteranceData:
        audio = np.concatenate(self._buffer) if len(self._buffer) > 1 else self._buffer[0]
        start_ms = int(self._start_sample / SAMPLE_RATE * 1000)
        end_ms = int((self._start_sample + self._buffer_samples) / SAMPLE_RATE * 1000)
        log.debug("Utterance: %d–%d ms (%d samples)", start_ms, end_ms, len(audio))
        return UtteranceData(audio=audio, start_ms=start_ms, end_ms=end_ms)

    def _reset(self) -> None:
        self._state = "idle"
        self._buffer = []
        self._buffer_samples = 0
        self._speech_samples = 0
        self._silence_counter = 0

    # emitted is set by _advance when endpoint detected, consumed by feed()
    _emitted: UtteranceData | None = None
