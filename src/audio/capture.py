"""Single-source audio capture via PyAudio callback.

Captures from a microphone or system loopback device and pushes
``AudioChunk`` objects to a thread-safe queue for downstream processing.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyaudio


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AudioChunk:
    """A slice of captured audio ready for transcription."""

    source: str          # "microphone" | "system_audio"
    data: np.ndarray     # float32, mono, shape (n_samples,)
    sample_rate: int     # Hz
    timestamp: float     # seconds since capture started


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def list_audio_devices() -> list[dict[str, Any]]:
    """Return all available input devices (mic + loopback).

    Each entry contains keys useful for a device-picker UI:
    ``index``, ``name``, ``host_api``, ``max_input_channels``,
    ``default_sample_rate``.
    """
    pa = pyaudio.PyAudio()
    devices: list[dict[str, Any]] = []
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append({
                    "index": i,
                    "name": info["name"],
                    "host_api": pa.get_host_api_info_by_index(
                        info["hostApi"]
                    )["name"],
                    "max_input_channels": info["maxInputChannels"],
                    "default_sample_rate": int(info["defaultSampleRate"]),
                })
    finally:
        pa.terminate()
    return devices


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------

class AudioCapture:
    """Captures audio from a single input device in a background thread.

    Pushes ``AudioChunk`` onto *queue* every *chunk_duration* seconds.
    """

    def __init__(
        self,
        source_label: str,
        device_index: int | None = None,
        sample_rate: int = 16000,
        chunk_duration: float = 1.0,
    ) -> None:
        self._source = source_label
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._chunk_duration = chunk_duration
        self._frames_per_chunk = int(sample_rate * chunk_duration)

        self._pa: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None
        self._queue: queue.Queue[AudioChunk] | None = None
        self._start_time: float = 0.0

    # -- public ----------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._stream is not None and self._stream.is_active()

    def start(self, target_queue: queue.Queue[AudioChunk]) -> None:
        """Begin capturing and push chunks to *target_queue*."""
        if self.is_running:
            return

        self._queue = target_queue
        self._start_time = time.monotonic()
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._sample_rate,
            input=True,
            input_device_index=self._device_index,
            frames_per_buffer=self._frames_per_chunk,
            stream_callback=self._on_audio,
        )
        self._stream.start_stream()

    def stop(self) -> None:
        """Stop capturing and release resources."""
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
        self._queue = None

    # -- callback --------------------------------------------------------

    def _on_audio(
        self,
        in_data: bytes | None,
        frame_count: int,
        time_info: dict,
        status_flags: int,
    ) -> tuple[bytes | None, int]:
        if in_data and self._queue is not None:
            samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            elapsed = time.monotonic() - self._start_time
            chunk = AudioChunk(
                source=self._source,
                data=samples,
                sample_rate=self._sample_rate,
                timestamp=elapsed,
            )
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                pass  # drop frame rather than block the audio thread

        return (None, pyaudio.paContinue)
