"""Orchestrates microphone and system-audio capture sources.

Provides a single audio queue that the transcription engine consumes.
Sources can be toggled independently at runtime.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from enum import Enum, auto

from src.audio.capture import AudioCapture, AudioChunk, list_audio_devices


class SourceState(Enum):
    STOPPED = auto()
    RUNNING = auto()


@dataclass
class SourceInfo:
    label: str
    device_index: int | None
    state: SourceState


class AudioManager:
    """Manages 0–2 audio capture sources feeding a single output queue."""

    def __init__(self) -> None:
        self._queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=256)
        self._mic: AudioCapture | None = None
        self._sys: AudioCapture | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # public properties
    # ------------------------------------------------------------------

    @property
    def audio_queue(self) -> queue.Queue[AudioChunk]:
        """Thread-safe queue that receives ``AudioChunk`` objects."""
        return self._queue

    @property
    def microphone_state(self) -> SourceState:
        return SourceState.RUNNING if (self._mic and self._mic.is_running) else SourceState.STOPPED  # fmt: skip

    @property
    def system_audio_state(self) -> SourceState:
        return SourceState.RUNNING if (self._sys and self._sys.is_running) else SourceState.STOPPED  # fmt: skip

    # ------------------------------------------------------------------
    # source control
    # ------------------------------------------------------------------

    def start_microphone(self, device_index: int | None = None) -> None:
        with self._lock:
            if self._mic and self._mic.is_running:
                return
            self._mic = AudioCapture(source_label="microphone", device_index=device_index)
            self._mic.start(self._queue)

    def stop_microphone(self) -> None:
        with self._lock:
            if self._mic:
                self._mic.stop()
                self._mic = None

    def start_system_audio(self, device_index: int | None = None) -> None:
        with self._lock:
            if self._sys and self._sys.is_running:
                return
            self._sys = AudioCapture(source_label="system_audio", device_index=device_index)
            self._sys.start(self._queue)

    def stop_system_audio(self) -> None:
        with self._lock:
            if self._sys:
                self._sys.stop()
                self._sys = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def stop_all(self) -> None:
        """Stop all capture sources and clear the queue."""
        with self._lock:
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

    @property
    def is_any_running(self) -> bool:
        return self.microphone_state == SourceState.RUNNING or self.system_audio_state == SourceState.RUNNING  # fmt: skip


# ------------------------------------------------------------------
# utility
# ------------------------------------------------------------------

def list_input_devices() -> list[dict]:
    """Convenience re-export for the UI layer."""
    return list_audio_devices()
