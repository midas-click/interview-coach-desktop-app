"""Tests for transcription engine and segment builder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.transcription.engine import TranscriptionEngine, TranscriptionSegment
from src.transcription.segment_builder import SegmentBuilder, _split_sentences, _words


# ---------------------------------------------------------------------------
# _words
# ---------------------------------------------------------------------------

def test_words_strips_punctuation() -> None:
    assert _words("Hello, world!") == ["hello", "world"]
    assert _words("I have ten years of experience.") == ["i", "have", "ten", "years", "of", "experience"]


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------

def test_split_sentences_single() -> None:
    assert _split_sentences("Hello world") == ["Hello world"]


def test_split_sentences_multiple() -> None:
    result = _split_sentences("Hello. World! How are you?")
    assert len(result) == 3
    assert result[0].startswith("Hello.")


def test_split_sentences_no_trailing_space() -> None:
    """Period at end-of-string without following whitespace is not a split."""
    assert _split_sentences("Done.") == ["Done."]


# ---------------------------------------------------------------------------
# SegmentBuilder
# ---------------------------------------------------------------------------

def test_segment_builder_merges_partials() -> None:
    sb = SegmentBuilder()
    assert sb.feed(TranscriptionSegment("Hello", 0.0, 1.0, 0.9)) == []
    assert sb.feed(TranscriptionSegment(" world.", 1.0, 2.0, 0.85)) == []
    chunks = sb.flush()
    assert len(chunks) == 1
    assert chunks[0].text == "Hello  world."  # whisper artifacts preserved


def test_segment_builder_dedup() -> None:
    sb = SegmentBuilder()
    sb.feed(TranscriptionSegment("I like pizza very much.", 0.0, 2.0, 0.9))
    # mostly overlapping text should be dropped
    assert sb.feed(TranscriptionSegment("pizza very much", 1.0, 2.5, 0.9)) == []
    chunks = sb.flush()
    assert len(chunks) == 1
    assert "pizza" in chunks[0].text


def test_segment_builder_sentence_boundary_emits() -> None:
    sb = SegmentBuilder()
    sb.feed(TranscriptionSegment("First sentence.", 0.0, 2.0, 0.9))
    chunks = sb.feed(TranscriptionSegment("Second sentence.", 2.0, 4.0, 0.9))
    assert len(chunks) == 1
    assert chunks[0].text == "First sentence."


def test_segment_builder_flush_empty() -> None:
    sb = SegmentBuilder()
    assert sb.flush() == []


def test_segment_builder_respects_timestamps() -> None:
    sb = SegmentBuilder()
    sb.feed(TranscriptionSegment("Hello.", 1.0, 2.0, 0.9))
    # gap > 0.5s → pending is finalised at its own boundaries
    chunks = sb.feed(TranscriptionSegment("World.", 3.0, 4.0, 0.9))
    assert len(chunks) == 1
    assert chunks[0].start_time == 1.0
    assert chunks[0].end_time == 2.0  # not extended across gap


def test_segment_builder_contiguous_extends_timestamps() -> None:
    sb = SegmentBuilder()
    sb.feed(TranscriptionSegment("Partial", 1.0, 2.0, 0.9))
    chunks = sb.feed(TranscriptionSegment(" sentence.", 2.0, 3.0, 0.9))
    # flush to get the merged result
    chunks = sb.flush()
    assert len(chunks) == 1
    assert chunks[0].start_time == 1.0
    assert chunks[0].end_time == 3.0  # extended across contiguous segment


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


def test_engine_start_stop(mock_whisper: MagicMock) -> None:
    engine = TranscriptionEngine(model_name="base")
    from queue import Queue
    from src.audio.capture import AudioChunk

    q: Queue[AudioChunk] = Queue()
    received: list[TranscriptionSegment] = []

    engine.start(q, on_segment=received.append)
    assert engine._running

    engine.stop()
    assert not engine._running


def test_engine_flushes_remaining_on_stop(mock_whisper: MagicMock) -> None:
    engine = TranscriptionEngine(model_name="base")
    from queue import Queue
    from src.audio.capture import AudioChunk

    q: Queue[AudioChunk] = Queue()
    mock_whisper.return_value.transcribe.return_value = (
        [FakeSegment("Final words.", 0.0, 1.0, -0.5)],
        None,
    )

    # feed one chunk and immediately stop
    data = np.zeros(8000, dtype=np.float32)
    q.put(AudioChunk(source="test", data=data, sample_rate=16000, timestamp=0.0))
    engine.start(q, on_segment=lambda s: None)
    import time
    time.sleep(0.3)  # let worker pick up the chunk

    segments = engine.stop()
    assert len(segments) == 1
    assert segments[0].text == "Final words."
