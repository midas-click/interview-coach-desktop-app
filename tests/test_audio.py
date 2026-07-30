"""Tests for audio capture and manager modules."""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.audio.capture import AudioCapture, AudioChunk, list_microphones, list_loopback_devices, LoopbackCapture
from src.audio.manager import AudioManager, SourceState


# ---------------------------------------------------------------------------
# list_audio_devices
# ---------------------------------------------------------------------------

def test_list_microphones() -> None:
    devices = list_microphones()
    assert isinstance(devices, list)
    for d in devices:
        assert "index" in d
        assert d["max_input_channels"] > 0


def test_list_loopback_devices() -> None:
    devices = list_loopback_devices()
    assert isinstance(devices, list)
    for d in devices:
        assert "index" in d
        assert d["max_output_channels"] > 0


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
# AudioCapture (mock sounddevice)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_sd() -> MagicMock:
    with patch("src.audio.capture.sd.InputStream") as mock_stream_cls, \
         patch("src.audio.capture.sd.query_devices") as mock_query:
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_stream_cls.return_value = mock_stream
        mock_query.return_value = {"max_input_channels": 1, "default_samplerate": 44100}
        yield mock_stream_cls


def test_capture_start_stop(mock_sd: MagicMock) -> None:
    cap = AudioCapture(source_label="microphone")
    q: queue.Queue[AudioChunk] = queue.Queue()

    assert not cap.is_running
    cap.start(q)
    assert cap.is_running
    cap.stop()
    assert not cap.is_running


def test_capture_callback_buffers_and_emits(mock_sd: MagicMock) -> None:
    cap = AudioCapture(source_label="microphone", device_index=1)
    q: queue.Queue[AudioChunk] = queue.Queue()
    cap.start(q)
    cap._sample_rate = 16000  # override for predictable test

    # First chunk — too small, should be buffered
    small = np.zeros((8000, 2), dtype=np.float32)  # 0.5s at 16kHz
    cap._on_audio(small, 8000, None, 0)
    assert q.empty()  # still buffered

    # Second chunk — together they exceed chunk_duration (1.0s)
    cap._on_audio(small, 8000, None, 0)
    chunk = q.get(timeout=1)
    assert chunk.source == "microphone"
    assert chunk.data.shape == (16000,)
    assert chunk.data.dtype == np.float32

    cap.stop()


# ---------------------------------------------------------------------------
# AudioManager
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_capture() -> MagicMock:
    with patch("src.audio.manager.AudioCapture") as mock_mic, \
         patch("src.audio.manager.LoopbackCapture") as mock_sys:
        mock_mic.return_value.is_running = True
        mock_sys.return_value.is_running = True
        yield {"mic": mock_mic, "sys": mock_sys}


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

    mock_mic = mock_capture["mic"]
    mock_mic.assert_called_once_with(source_label="microphone", device_index=2)
    mock_mic.return_value.start.assert_called_once_with(mgr.audio_queue)

    mgr.stop_microphone()
    assert mgr.microphone_state == SourceState.STOPPED
    mock_mic.return_value.stop.assert_called_once()


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
    assert mock_capture["mic"].call_count == 1


def test_manager_stop_clears_queue(mock_capture: MagicMock) -> None:
    mgr = AudioManager()
    data = np.zeros(100, dtype=np.float32)
    mgr.audio_queue.put(AudioChunk(source="test", data=data, sample_rate=16000, timestamp=0.0))
    assert not mgr.audio_queue.empty()

    mgr.stop_all()
    assert mgr.audio_queue.empty()
