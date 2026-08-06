"""Tests for transcription engine and worker."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.transcription.engine import TranscriptionEngine, TranscriptionSegment


# ---------------------------------------------------------------------------
# TranscriptionEngine (mock faster-whisper)
# ---------------------------------------------------------------------------

class FakeSegment:
    def __init__(self, text: str, start: float, end: float, avg_logprob: float):
        self.text = text
        self.start = start
        self.end = end
        self.avg_logprob = avg_logprob


@pytest.fixture
def mock_whisper() -> MagicMock:
    with patch("src.transcription.engine.WhisperModel") as mock:
        yield mock


def test_engine_transcribe_wav(mock_whisper: MagicMock) -> None:
    mock_whisper.return_value.transcribe.return_value = (
        [FakeSegment("Hello world.", 0.0, 1.0, -0.3)],
        None,
    )
    engine = TranscriptionEngine(model_name="base")
    segments = engine.transcribe("test.wav")
    assert len(segments) == 1
    assert segments[0].text == "Hello world."
    assert segments[0].confidence == -0.3


def test_engine_filters_low_confidence(mock_whisper: MagicMock) -> None:
    mock_whisper.return_value.transcribe.return_value = (
        [FakeSegment("noise", 0.0, 1.0, -1.5)],
        None,
    )
    engine = TranscriptionEngine(model_name="base")
    segments = engine.transcribe("test.wav")
    assert len(segments) == 0


def test_engine_filters_short_text(mock_whisper: MagicMock) -> None:
    mock_whisper.return_value.transcribe.return_value = (
        [FakeSegment("Hi.", 0.0, 1.0, -0.3)],
        None,
    )
    engine = TranscriptionEngine(model_name="base")
    segments = engine.transcribe("test.wav")
    assert len(segments) == 0


# ---------------------------------------------------------------------------
# TranscriptionWorker (integration test with real DB)
# ---------------------------------------------------------------------------

@pytest.fixture
def worker_db(tmp_path: Path):
    from src.storage.repository import Repository
    db_path = tmp_path / "test.db"
    repo = Repository(db_path)

    async def _init():
        await repo.init()
    asyncio.run(_init())
    return repo


def test_worker_processes_utterance(worker_db, tmp_path: Path, mock_whisper: MagicMock) -> None:
    from src.transcription.worker import TranscriptionWorker
    import wave
    import numpy as np

    mock_whisper.return_value.transcribe.return_value = (
        [FakeSegment("Tell me about yourself.", 0.0, 2.0, -0.2)],
        None,
    )

    # Create a test WAV
    wav_path = tmp_path / "test.wav"
    audio = np.zeros(16000, dtype=np.int16)
    audio[8000:12000] = 1000  # some fake signal
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio.tobytes())

    # Insert interview + utterance
    async def _setup():
        await worker_db.create_interview("test-1")
        conn = __import__("sqlite3").connect(worker_db.db_path)
        conn.execute(
            "INSERT INTO utterances (interview_id, speaker, start_ms, end_ms, audio_path, status) "
            "VALUES ('test-1', 'interviewer', 0, 2000, ?, 'queued')",
            (str(wav_path),),
        )
        conn.commit()
        conn.close()

    asyncio.run(_setup())

    engine = TranscriptionEngine(model_name="base")
    worker = TranscriptionWorker(worker_db, engine)

    worker.start()

    # Wait for processing
    import time
    deadline = time.time() + 10
    while time.time() < deadline:
        utterance = asyncio.run(worker_db.next_queued())
        if utterance is None:
            # No more queued — check if completed
            break
        time.sleep(0.3)

    worker.stop()

    # Verify utterance was processed
    async def _verify():
        utterances = await worker_db.get_utterances("test-1")
        assert len(utterances) == 1
        assert utterances[0].transcript == "Tell me about yourself."
        assert utterances[0].confidence == -0.2
        assert not wav_path.exists()  # WAV was deleted

    asyncio.run(_verify())
