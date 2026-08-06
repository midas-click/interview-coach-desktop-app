"""Tests for audio capture and manager modules."""

from __future__ import annotations

import queue
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.audio.capture import AudioCapture, AudioChunk, list_microphones, list_loopback_devices


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
    cap._sample_rate = 16000

    small = np.zeros((8000, 2), dtype=np.float32)
    cap._on_audio(small, 8000, None, 0)
    assert q.empty()

    cap._on_audio(small, 8000, None, 0)
    chunk = q.get(timeout=1)
    assert chunk.source == "microphone"
    assert chunk.data.shape == (16000,)
    assert chunk.data.dtype == np.float32

    cap.stop()
