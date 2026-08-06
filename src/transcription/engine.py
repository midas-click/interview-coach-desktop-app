"""faster-whisper transcription engine — one-shot WAV transcription.

Loads the model once. Single ``transcribe()`` method that takes a WAV
file path and returns a list of segments.  No streaming, no buffers,
no threading — the worker thread owns concurrency.
"""

from __future__ import annotations

from dataclasses import dataclass

from faster_whisper import WhisperModel

from src.logger.logger import get_logger

log = get_logger(__name__)

CONFIDENCE_THRESHOLD = -0.8
MIN_WORDS = 2


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    """A transcribed speech segment."""

    text: str
    start: float
    end: float
    confidence: float


class TranscriptionEngine:
    """One-shot transcription via faster-whisper."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
    ) -> None:
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
        log.info("Whisper model loaded: %s on %s (%s)", model_name, device, compute_type)

    def transcribe(self, wav_path: str) -> list[TranscriptionSegment]:
        """Transcribe a single WAV file. Returns segments sorted by start time."""
        raw_segments, _info = self._model.transcribe(
            wav_path,
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
            if seg.avg_logprob < CONFIDENCE_THRESHOLD:
                continue
            text = seg.text.strip()
            if len(text.split()) < MIN_WORDS:
                continue
            results.append(TranscriptionSegment(
                text=text,
                start=round(seg.start, 2),
                end=round(seg.end, 2),
                confidence=round(seg.avg_logprob, 3),
            ))
        return results
