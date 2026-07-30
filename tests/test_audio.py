"""Tests for audio capture and manager modules."""

from __future__ import annotations

import queue
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.audio.capture import AudioCapture, AudioChunk, list_audio_devices
from src.audio.manager import AudioManager, SourceState


# ---------------------------------------------------------------------------
# list_audio_devices
# ---------------------------------------------------------------------------

def test_list_devices_returns_list() -> None:
    devices = list_audio_devices()
    assert isinstance(devices, list)
    for d in devices:
        assert "index" in d
        assert "name" in d
        assert "max_input_channels" in d


# ---------------------------------------------------------------------------
# AudioChunk
# ---------------------------------------------------------------------------

def test_audio_chunk_immutable() -> None:
    data = np.zeros(16000, dtype=np.float32)
    chunk = AudioChunk(source="microphone", data=data, sample_rate=16000, timestamp=1.5)
    assert chunk.source == "microphone"
    assert chunk.sample_rate == 16000
    assert chunk.timestamp == 1.5
    with pytest.raises(Exception):
        chunk.source = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AudioCapture (mock PyAudio)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_pyaudio() -> MagicMock:
    with patch("src.audio.capture.pyaudio.PyAudio") as mock:
        mock_stream = MagicMock()
        mock_stream.is_active.return_value = True
        mock.return_value.open.return_value = mock_stream
        yield mock


def test_capture_start_stop(mock_pyaudio: MagicMock) -> None:
    cap = AudioCapture(source_label="microphone")
    q: queue.Queue[AudioChunk] = queue.Queue()

    assert not cap.is_running
    cap.start(q)
    assert cap.is_running
    cap.stop()
    assert not cap.is_running


def test_capture_callback_pushes_chunk(mock_pyaudio: MagicMock) -> None:
    cap = AudioCapture(source_label="system_audio", device_index=1)
    q: queue.Queue[AudioChunk] = queue.Queue()
    cap.start(q)

    # simulate PyAudio callback with 0.5s of 16-bit silence
    frame_count = 8000
    silence = b"\x00" * (frame_count * 2)  # int16 = 2 bytes per sample
    cap._on_audio(silence, frame_count, {}, 0)

    chunk = q.get(timeout=1)
    assert chunk.source == "system_audio"
    assert chunk.sample_rate == 16000
    assert chunk.data.shape == (frame_count,)
    assert chunk.data.dtype == np.float32

    cap.stop()


def test_callback_drops_on_full_queue(mock_pyaudio: MagicMock) -> None:
    cap = AudioCapture(source_label="microphone")
    q: queue.Queue[AudioChunk] = queue.Queue(maxsize=1)
    cap.start(q)

    frame_count = 100
    silence = b"\x00" * (frame_count * 2)

    # fill the queue
    cap._on_audio(silence, frame_count, {}, 0)
    assert q.full()

    # next callback should not deadlock
    cap._on_audio(silence, frame_count, {}, 0)
    cap.stop()


# ---------------------------------------------------------------------------
# AudioManager
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_capture() -> MagicMock:
    with patch("src.audio.manager.AudioCapture") as mock:
        mock.return_value.is_running = True
        yield mock


def test_manager_initial_state() -> None:
    mgr = AudioManager()
    assert mgr.microphone_state == SourceState.STOPPED
    assert mgr.system_audio_state == SourceState.STOPPED
    assert not mgr.is_any_running
    assert isinstance(mgr.audio_queue, queue.Queue)


def test_manager_start_stop_mic(mock_capture: MagicMock) -> None:
    mgr = AudioManager()
    mgr.start_microphone(device_index=2)
    assert mgr.microphone_state == SourceState.RUNNING
    assert mgr.is_any_running

    mock_capture.assert_called_once_with(source_label="microphone", device_index=2)
    mock_capture.return_value.start.assert_called_once_with(mgr.audio_queue)

    mgr.stop_microphone()
    assert mgr.microphone_state == SourceState.STOPPED
    mock_capture.return_value.stop.assert_called_once()


def test_manager_start_stop_system(mock_capture: MagicMock) -> None:
    mgr = AudioManager()
    mgr.start_system_audio()
    assert mgr.system_audio_state == SourceState.RUNNING

    mgr.stop_system_audio()
    assert mgr.system_audio_state == SourceState.STOPPED


def test_manager_both_sources(mock_capture: MagicMock) -> None:
    mgr = AudioManager()
    mgr.start_microphone()
    mgr.start_system_audio()
    assert mgr.microphone_state == SourceState.RUNNING
    assert mgr.system_audio_state == SourceState.RUNNING

    mgr.stop_all()
    assert mgr.microphone_state == SourceState.STOPPED
    assert mgr.system_audio_state == SourceState.STOPPED
    assert not mgr.is_any_running


def test_manager_double_start_is_idempotent(mock_capture: MagicMock) -> None:
    mgr = AudioManager()
    mgr.start_microphone()
    mgr.start_microphone()  # second call — should be no-op
    assert mock_capture.call_count == 1


def test_manager_stop_clears_queue(mock_capture: MagicMock) -> None:
    mgr = AudioManager()
    # push some junk into the queue
    data = np.zeros(100, dtype=np.float32)
    mgr.audio_queue.put(AudioChunk(source="test", data=data, sample_rate=16000, timestamp=0.0))
    assert not mgr.audio_queue.empty()

    mgr.stop_all()
    assert mgr.audio_queue.empty()
