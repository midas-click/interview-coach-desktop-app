"""Async SQLite repository for meetings and transcript chunks."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from src.storage.models import Meeting, MeetingStatus, TranscriptChunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    meeting_id      TEXT PRIMARY KEY,
    company_name    TEXT,
    interview_stage TEXT,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    ended_at        TEXT,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','active','finished'))
);

CREATE TABLE IF NOT EXISTS transcript_chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id    TEXT NOT NULL REFERENCES meetings(meeting_id),
    speaker       TEXT NOT NULL DEFAULT 'Unknown',
    start_time    REAL NOT NULL,
    end_time      REAL NOT NULL,
    confidence    REAL NOT NULL DEFAULT 0.0,
    text          TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_chunks_meeting ON transcript_chunks(meeting_id);
CREATE INDEX IF NOT EXISTS idx_chunks_time ON transcript_chunks(meeting_id, start_time);
"""


class Repository:
    """Async data access for meetings and transcript chunks.

    Uses WAL mode for non-blocking reads during active transcription.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = str(db_path)

    # ------------------------------------------------------------------
    # initialisation
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """Create tables and enable WAL mode. Idempotent."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA foreign_keys=ON")
            await db.executescript(_SCHEMA)
            await db.commit()

    # ------------------------------------------------------------------
    # meeting lifecycle
    # ------------------------------------------------------------------

    async def create_meeting(
        self,
        meeting_id: str,
        company_name: str | None = None,
        interview_stage: str | None = None,
    ) -> Meeting:
        meeting = Meeting(
            meeting_id=meeting_id,
            company_name=company_name,
            interview_stage=interview_stage,
            created_at=datetime.now(tz=timezone.utc),
        )
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO meetings (meeting_id, company_name, interview_stage, created_at, status) "
                "VALUES (?, ?, ?, ?, ?)",
                (meeting.meeting_id, meeting.company_name, meeting.interview_stage,
                 meeting.created_at.isoformat(), meeting.status.value),
            )
            await db.commit()
        return meeting

    async def _update_status(self, meeting_id: str, status: MeetingStatus) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        extra_sql = ""
        params: list = [status.value, meeting_id]

        if status == MeetingStatus.ACTIVE:
            extra_sql = ", started_at = COALESCE(started_at, ?)"
            params.insert(1, now)
        elif status == MeetingStatus.FINISHED:
            extra_sql = ", ended_at = COALESCE(ended_at, ?)"
            params.insert(1, now)

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                f"UPDATE meetings SET status = ?{extra_sql} WHERE meeting_id = ?",
                params,
            )
            await db.commit()

    async def start_meeting(self, meeting_id: str) -> None:
        await self._update_status(meeting_id, MeetingStatus.ACTIVE)

    async def finish_meeting(self, meeting_id: str) -> None:
        await self._update_status(meeting_id, MeetingStatus.FINISHED)

    async def get_meeting(self, meeting_id: str) -> Meeting | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM meetings WHERE meeting_id = ?", (meeting_id,)
            )
            row = await cursor.fetchone()
            return self._row_to_meeting(row) if row else None

    # ------------------------------------------------------------------
    # transcript chunks
    # ------------------------------------------------------------------

    async def insert_chunks(self, chunks: list[TranscriptChunk]) -> None:
        """Batch-insert transcript chunks."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.executemany(
                "INSERT INTO transcript_chunks (meeting_id, speaker, start_time, end_time, "
                "confidence, text) VALUES (?, ?, ?, ?, ?, ?)",
                [(c.meeting_id, c.speaker, c.start_time, c.end_time, c.confidence, c.text)
                 for c in chunks],
            )
            await db.commit()

    async def get_chunks(self, meeting_id: str) -> list[TranscriptChunk]:
        """Return all chunks for a meeting, ordered by start_time."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM transcript_chunks WHERE meeting_id = ? ORDER BY start_time",
                (meeting_id,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_chunk(r) for r in rows]

    async def export_transcript(self, meeting_id: str) -> dict:
        """Return the full transcript payload matching the spec JSON format."""
        meeting = await self.get_meeting(meeting_id)
        chunks = await self.get_chunks(meeting_id)

        return {
            "meetingId": meeting_id,
            "companyName": meeting.company_name if meeting else None,
            "interviewStage": meeting.interview_stage if meeting else None,
            "createdAt": meeting.created_at.isoformat() if meeting else None,
            "language": "en",
            "transcript": [
                {
                    "speaker": c.speaker,
                    "start": c.start_time,
                    "end": c.end_time,
                    "confidence": c.confidence,
                    "text": c.text,
                }
                for c in chunks
            ],
        }

    async def export_txt(self, meeting_id: str) -> str:
        """Return the transcript as a plain-text document.

        Format::

            [00:00:01] Unknown: Tell me about yourself.
            [00:00:05] Unknown: I have 10 years experience.
        """
        meeting = await self.get_meeting(meeting_id)
        chunks = await self.get_chunks(meeting_id)

        lines: list[str] = []
        if meeting:
            lines.append(f"Meeting: {meeting_id}")
            if meeting.company_name:
                lines.append(f"Company: {meeting.company_name}")
            if meeting.interview_stage:
                lines.append(f"Stage: {meeting.interview_stage}")
            lines.append(f"Date: {meeting.created_at.strftime('%Y-%m-%d %H:%M UTC')}")
            lines.append("")

        for c in chunks:
            ts = self._format_timestamp(c.start_time)
            lines.append(f"[{ts}] {c.speaker}: {c.text}")

        return "\n".join(lines)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_meeting(row: aiosqlite.Row) -> Meeting:
        def _dt(val: str | None) -> datetime | None:
            return datetime.fromisoformat(val) if val else None

        return Meeting(
            meeting_id=row["meeting_id"],
            company_name=row["company_name"],
            interview_stage=row["interview_stage"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=_dt(row["started_at"]),
            ended_at=_dt(row["ended_at"]),
            status=MeetingStatus(row["status"]),
        )

    @staticmethod
    def _row_to_chunk(row: aiosqlite.Row) -> TranscriptChunk:
        def _dt(val: str | None) -> datetime | None:
            return datetime.fromisoformat(val) if val else None

        return TranscriptChunk(
            id=row["id"],
            meeting_id=row["meeting_id"],
            speaker=row["speaker"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            confidence=row["confidence"],
            text=row["text"],
            created_at=_dt(row["created_at"]),
        )
