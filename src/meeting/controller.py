"""Interview lifecycle orchestrator.

Coordinates audio capture, background transcription, and
database persistence.  Does NOT perform transcription itself.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from src.audio.manager import AudioManager
from src.logger.logger import get_logger
from src.transcription.engine import TranscriptionEngine
from src.transcription.worker import TranscriptionWorker

log = get_logger(__name__)

if TYPE_CHECKING:
    from src.config.settings import Settings
    from src.storage.repository import Repository


class MeetingController:
    """Manages one interview at a time.

    Callbacks
    ---------
    ``on_status_change(status: str)``
        Called from the asyncio event loop when the meeting status transitions.
    """

    def __init__(self, settings: Settings, repository: Repository) -> None:
        self._settings = settings
        self._repo = repository

        self._engine = TranscriptionEngine(
            model_name=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )

        self._audio = AudioManager(
            db_path=self._repo.db_path,
            temp_dir=settings.temp_dir,
            vad_threshold=settings.vad_threshold,
            vad_silence_ms=settings.vad_silence_ms,
            vad_min_speech_ms=settings.vad_min_speech_ms,
        )

        self._worker = TranscriptionWorker(self._repo, self._engine)

        self._interview_id: str | None = None

        # callbacks
        self.on_status_change: StatusCallback | None = None

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    @property
    def interview_id(self) -> str | None:
        return self._interview_id

    @property
    def audio(self) -> AudioManager:
        return self._audio

    async def create_interview(
        self,
        company: str | None = None,
        stage: str | None = None,
    ) -> str:
        """Create a new interview row and return its ID."""
        interview_id = str(uuid.uuid4())
        await self._repo.create_interview(interview_id, company, stage)
        self._interview_id = interview_id
        return interview_id

    async def start(
        self,
        mic_device: int | None = None,
        sys_device: int | None = None,
    ) -> None:
        """Begin audio capture and background transcription."""
        if self._interview_id is None:
            raise RuntimeError("No interview created — call create_interview() first")

        self._audio.start(self._interview_id, mic_device, sys_device)
        self._worker.start()
        await self._repo.start_interview(self._interview_id)
        self._notify_status("Actively listening…")
        log.info("Interview started: %s", self._interview_id)

    async def finish(self) -> dict:
        """Stop capture, drain transcription queue, finalise, return JSON export."""
        log.info("Finishing interview %s", self._interview_id)

        # 1. Stop audio capture — flushes final utterances as queued
        self._notify_status("Stopping capture…")
        self._audio.stop_all()

        # 2. Wait for all utterances to be transcribed
        self._notify_status("Transcribing remaining audio…")
        while True:
            pending = await self._repo.count_pending(self._interview_id)
            if pending == 0:
                log.info("All utterances transcribed — queue empty")
                break
            log.info("Waiting for transcription: %d utterance(s) remaining", pending)
            await asyncio.sleep(0.5)

        # 3. Stop worker
        self._worker.stop()

        # 4. Finalise interview
        await self._repo.finish_interview(self._interview_id)

        # 5. Export
        self._notify_status("Generating export…")
        export = await self.export_current()
        txt = await self._repo.export_txt(self._interview_id)
        self._save_txt_local(txt)

        self._notify_status("Ready")
        return export

    async def export_current(self) -> dict:
        """Export the current interview's transcript as JSON."""
        if self._interview_id is None:
            raise RuntimeError("No interview to export")
        return await self._repo.export_json(
            self._interview_id,
            model=f"whisper-{self._settings.whisper_model}",
            language="en",
        )

    def _save_txt_local(self, text: str) -> None:
        if not text.strip():
            return
        dir_path = self._settings.output_dir / (self._interview_id or "unknown")
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "transcript.txt").write_text(text, encoding="utf-8")

    def _notify_status(self, status: str) -> None:
        if self.on_status_change:
            self.on_status_change(status)


# ---------------------------------------------------------------------------
# types
# ---------------------------------------------------------------------------

from collections.abc import Callable

StatusCallback = Callable[[str], None]
