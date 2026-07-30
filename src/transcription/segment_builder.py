"""Merges raw whisper segments into clean, non-overlapping sentences.

Handles the common streaming-transcription artifacts:
* overlapping re-transcription of the same audio window
* partial sentences that span segment boundaries
* trailing filler words from VAD boundary imprecision
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.storage.models import TranscriptChunk
from src.transcription.engine import TranscriptionSegment

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_PUNCT_STRIP = re.compile(r"[^\w\s]")


def _words(text: str) -> list[str]:
    """Lower-case token list with punctuation stripped."""
    return _PUNCT_STRIP.sub("", text.lower()).split()


# ---------------------------------------------------------------------------
# sentence splitting
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split *text* on sentence boundaries, keeping the punctuation."""
    parts = _SENTENCE_END.split(text)
    # re-attach trailing punctuation that got consumed by the split
    result: list[str] = []
    punct: list[str] = []
    for m in _SENTENCE_END.finditer(text):
        punct.append(m.group())
    for i, part in enumerate(parts):
        if i < len(punct):
            result.append(part + punct[i])
        else:
            result.append(part)
    return [s for s in result if s.strip()]


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


@dataclass
class _Pending:
    text: str
    start: float
    end: float
    confidence: float
    speaker: str = "Unknown"


@dataclass
class SegmentBuilder:
    """Accumulates partial transcription results and emits finalised chunks."""

    _pending: _Pending | None = field(default=None, init=False)

    def feed(self, segment: TranscriptionSegment) -> list[TranscriptChunk]:
        """Process a new segment. Returns any chunks that are now final."""
        chunks: list[TranscriptChunk] = []

        # dedup: skip if fully contained in previous
        if self._pending and self._overlap_ratio(segment.text, self._pending.text) > 0.8:
            self._pending.end = segment.end
            return chunks

        # gap detected — finalise old pending before starting fresh
        if self._pending and segment.start > self._pending.end + 0.5:
            chunks.extend(self._emit_pending())
            self._pending = _Pending(
                text=segment.text,
                start=segment.start,
                end=segment.end,
                confidence=segment.confidence,
                speaker=segment.speaker,
            )
            return chunks

        # merge with pending
        merged = self._merge(segment)
        sentences = _split_sentences(merged.text)

        if len(sentences) <= 1:
            # keep as pending – not yet a complete sentence
            self._pending = _Pending(
                text=sentences[0] if sentences else merged.text,
                start=merged.start,
                end=merged.end,
                confidence=merged.confidence,
                speaker=merged.speaker,
            )
            return chunks

        # emit all but the last sentence (which may be partial)
        for sent in sentences[:-1]:
            chunks.append(self._make_chunk(sent, merged.start, merged.end, merged.confidence, merged.speaker))

        # keep the last partial sentence
        last = sentences[-1]
        self._pending = _Pending(
            text=last,
            start=merged.start,
            end=merged.end,
            confidence=merged.confidence,
            speaker=merged.speaker,
        )
        return chunks

    def _emit_pending(self) -> list[TranscriptChunk]:
        """Convert current pending to a chunk and clear it."""
        if self._pending is None or not self._pending.text.strip():
            self._pending = None
            return []
        chunk = self._make_chunk(
            self._pending.text,
            self._pending.start,
            self._pending.end,
            self._pending.confidence,
            self._pending.speaker,
        )
        self._pending = None
        return [chunk]

    def flush(self) -> list[TranscriptChunk]:
        """Emit whatever is still pending (called on stop)."""
        return self._emit_pending()

    # ----------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------

    def _merge(self, seg: TranscriptionSegment) -> _Pending:
        if self._pending is None:
            return _Pending(text=seg.text, start=seg.start, end=seg.end, confidence=seg.confidence, speaker=seg.speaker)

        # if timestamps overlap, extend
        if seg.start <= self._pending.end + 0.3:
            return _Pending(
                text=f"{self._pending.text} {seg.text}".strip(),
                start=self._pending.start,
                end=seg.end,
                confidence=(self._pending.confidence + seg.confidence) / 2,
                speaker=self._pending.speaker,
            )

        # gap — treat as new sentence
        return _Pending(text=seg.text, start=seg.start, end=seg.end, confidence=seg.confidence, speaker=seg.speaker)

    @staticmethod
    def _overlap_ratio(new_text: str, old_text: str) -> float:
        """Fraction of *new_text* already present in *old_text*."""
        if not old_text or not new_text:
            return 0.0
        words_new = set(_words(new_text))
        words_old = set(_words(old_text))
        if not words_new:
            return 0.0
        return len(words_new & words_old) / len(words_new)

    @staticmethod
    def _make_chunk(text: str, start: float, end: float, confidence: float, speaker: str) -> TranscriptChunk:
        return TranscriptChunk(
            meeting_id="",  # filled by caller
            speaker=speaker,
            start_time=round(start, 2),
            end_time=round(end, 2),
            confidence=round(confidence, 3),
            text=text.strip(),
        )
