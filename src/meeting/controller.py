"""Meeting lifecycle orchestrator.

Coordinates audio capture, transcription, segment merging, and
periodic persistence to SQLite.  The controller itself is
async-friendly but callbacks from worker threads are bridged
via a simple lock.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import TYPE_CHECKING

from src.audio.manager import AudioManager
from src.logger.logger import get_logger
from src.storage.models import TranscriptChunk
from src.transcription.engine import (
    TranscriptionEngine,
    TranscriptionSegment,
)
from src.transcription.segment_builder import SegmentBuilder

log = get_logger(__name__)

if TYPE_CHECKING:
    from src.config.settings import Settings
    from src.storage.repository import Repository


class MeetingController:
    """Manages one interview meeting at a time.

    Callbacks
    ---------
    ``on_status_change(status: str)``
        Called from the asyncio event loop whenever the meeting
        status transitions.
    ``on_chunks_persisted(chunks: list[TranscriptChunk])``
        Called after a batch of chunks has been written to the
        database (useful for updating the live transcript UI).
    """

    def __init__(self, settings: Settings, repository: Repository) -> None:
        self._settings = settings
        self._repo = repository

        self._audio = AudioManager()
        self._engine = TranscriptionEngine(
            model_name=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        self._segment_builder = SegmentBuilder()

        # current meeting state
        self._meeting_id: str | None = None
        self._session_start: float = 0.0       # monotonic time when capture (re-)started
        self._elapsed_offset: float = 0.0       # accumulated seconds before current session

        # pending chunks waiting for periodic DB flush
        self._lock = threading.Lock()
        self._pending_chunks: list[TranscriptChunk] = []

        # background flush task
        self._flush_task: asyncio.Task[None] | None = None
        self._flush_stop: asyncio.Event | None = None

        # callbacks
        self.on_status_change: StatusCallback | None = None
        self.on_chunks_persisted: ChunksCallback | None = None

    # ------------------------------------------------------------------
    # public properties
    # ------------------------------------------------------------------

    @property
    def meeting_id(self) -> str | None:
        return self._meeting_id

    @property
    def is_active(self) -> bool:
        return self._audio.is_any_running

    @property
    def microphone_active(self) -> bool:
        return self._audio.microphone_state.name == "RUNNING"

    @property
    def system_audio_active(self) -> bool:
        return self._audio.system_audio_state.name == "RUNNING"

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def create_meeting(
        self,
        company_name: str | None = None,
        interview_stage: str | None = None,
    ) -> str:
        """Create a new meeting row and return its ID."""
        meeting_id = uuid.uuid4().hex[:12]
        await self._repo.create_meeting(meeting_id, company_name, interview_stage)
        self._meeting_id = meeting_id
        return meeting_id

    async def start(
        self,
        mic_device: int | None = None,
        sys_device: int | None = None,
    ) -> None:
        """Begin capturing audio and transcribing."""
        if self._meeting_id is None:
            raise RuntimeError("No meeting created — call create_meeting() first")

        # Always start microphone unless explicitly asked not to.
        # Passing a device index of None uses the system default.
        self._audio.start_microphone(mic_device)
        if sys_device is not None:
            try:
                self._audio.start_system_audio(sys_device)
            except Exception as exc:
                log.warning("System audio device %d failed: %s", sys_device, exc)

        log.info(
            "Starting transcription — mic=%s, sys=%s",
            "default" if mic_device is None else f"device {mic_device}",
            "off" if sys_device is None else f"device {sys_device}",
        )

        self._elapsed_offset = 0.0
        self._session_start = time.monotonic()
        self._engine.start(self._audio.audio_queue, self._on_transcription_segment)

        self._flush_stop = asyncio.Event()
        self._flush_task = asyncio.create_task(self._flush_loop())

        await self._repo.start_meeting(self._meeting_id)
        self._notify_status("active")

    async def pause(self) -> None:
        """Pause capture and transcription, persist everything."""
        log.info("Pausing meeting %s", self._meeting_id)
        self._audio.stop_all()
        remaining = self._engine.stop()
        for seg in remaining:
            self._on_transcription_segment(seg)

        # finalise segment builder
        with self._lock:
            chunks = self._segment_builder.flush()
            for c in chunks:
                c.meeting_id = self._meeting_id  # type: ignore[assignment]
            self._pending_chunks.extend(chunks)

        await self._persist_pending()
        self._stop_flush_loop()
        await self._repo.pause_meeting(self._meeting_id)  # type: ignore[arg-type]
        self._notify_status("paused")
        log.info("Meeting %s paused — %d chunks persisted", self._meeting_id, len(chunks))

    async def resume(
        self,
        mic_device: int | None = None,
        sys_device: int | None = None,
    ) -> None:
        """Resume after a pause, accumulating elapsed time offset."""
        log.info("Resuming meeting %s", self._meeting_id)
        self._elapsed_offset += time.monotonic() - self._session_start

        if mic_device is not None:
            self._audio.start_microphone(mic_device)
        if sys_device is not None:
            try:
                self._audio.start_system_audio(sys_device)
            except Exception as exc:
                log.warning("System audio device %d failed: %s", sys_device, exc)

        self._session_start = time.monotonic()
        self._engine.start(self._audio.audio_queue, self._on_transcription_segment)

        self._flush_stop = asyncio.Event()
        self._flush_task = asyncio.create_task(self._flush_loop())

        await self._repo.resume_meeting(self._meeting_id)  # type: ignore[arg-type]
        self._notify_status("active")

    async def finish(self) -> dict:
        """Stop everything, finalise, export JSON + TXT, return JSON payload."""
        log.info("Finishing meeting %s", self._meeting_id)
        self._audio.stop_all()
        remaining = self._engine.stop()
        for seg in remaining:
            self._on_transcription_segment(seg)

        with self._lock:
            chunks = self._segment_builder.flush()
            for c in chunks:
                c.meeting_id = self._meeting_id  # type: ignore[assignment]
            self._pending_chunks.extend(chunks)

        await self._persist_pending()
        self._stop_flush_loop()
        await self._repo.finish_meeting(self._meeting_id)  # type: ignore[arg-type]
        self._notify_status("finished")

        export = await self._repo.export_transcript(self._meeting_id)  # type: ignore[arg-type]
        txt = await self._repo.export_txt(self._meeting_id)  # type: ignore[arg-type]
        self._save_txt_local(txt)
        return export

    async def export_current(self) -> dict:
        """Re-export the current meeting's transcript (for manual upload)."""
        if self._meeting_id is None:
            raise RuntimeError("No meeting to export")
        export = await self._repo.export_transcript(self._meeting_id)
        self._save_txt_local(await self._repo.export_txt(self._meeting_id))
        return export

    def _save_txt_local(self, text: str) -> None:
        if not text.strip():
            return
        dir_path = self._settings.output_dir / (self._meeting_id or "unknown")
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "transcript.txt").write_text(text, encoding="utf-8")

    # ------------------------------------------------------------------
    # crash recovery
    # ------------------------------------------------------------------

    async def recover_active_meeting(self) -> str | None:
        """Check for an unfinished meeting from a previous run.

        Returns the meeting ID if found, otherwise ``None``.
        The caller should decide whether to resume or finalise it.
        """
        meeting = await self._repo.get_active_meeting()
        if meeting is None:
            return None
        self._meeting_id = meeting.meeting_id
        return meeting.meeting_id

    # ------------------------------------------------------------------
    # transcription callback (worker thread)
    # ------------------------------------------------------------------

    def _on_transcription_segment(self, segment: TranscriptionSegment) -> None:
        """Called from the transcription worker thread.

        The engine already emits segments with capture-relative timestamps
        (AudioChunk.timestamp + whisper offset).  We only need to add the
        accumulated pause time so timestamps stay meeting-relative across
        pause/resume cycles.
        """
        log.debug("Transcription segment: \"%s\" [%.1f-%.1f]", segment.text, segment.start, segment.end)
        adjusted = TranscriptionSegment(
            text=segment.text,
            start=segment.start + self._elapsed_offset,
            end=segment.end + self._elapsed_offset,
            confidence=segment.confidence,
            speaker=segment.speaker,
        )
        with self._lock:
            chunks = self._segment_builder.feed(adjusted)
            for c in chunks:
                c.meeting_id = self._meeting_id  # type: ignore[assignment]
            self._pending_chunks.extend(chunks)
            if chunks:
                log.info("Emitted %d chunk(s): %s", len(chunks), [c.text[:50] for c in chunks])

    # ------------------------------------------------------------------
    # periodic persistence
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        """Persist pending chunks every 3 seconds while the meeting is active."""
        while not self._flush_stop.is_set():  # type: ignore[union-attr]
            try:
                await asyncio.wait_for(
                    self._flush_stop.wait(),  # type: ignore[union-attr]
                    timeout=3.0,
                )
                break
            except asyncio.TimeoutError:
                await self._persist_pending()

    async def _persist_pending(self) -> None:
        with self._lock:
            if not self._pending_chunks:
                return
            chunks = self._pending_chunks
            self._pending_chunks = []

        await self._repo.insert_chunks(chunks)
        log.info("Persisted %d chunk(s) to DB", len(chunks))
        if self.on_chunks_persisted:
            self.on_chunks_persisted(chunks)

    def _stop_flush_loop(self) -> None:
        if self._flush_stop:
            self._flush_stop.set()
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()

    def _notify_status(self, status: str) -> None:
        if self.on_status_change:
            self.on_status_change(status)


# ---------------------------------------------------------------------------
# types
# ---------------------------------------------------------------------------

from collections.abc import Callable

StatusCallback = Callable[[str], None]
ChunksCallback = Callable[[list[TranscriptChunk]], None]
