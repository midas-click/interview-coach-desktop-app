"""Async SQLite repository for interviews and utterances."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from src.logger.logger import get_logger
from src.storage.models import Interview, InterviewStatus, Utterance, UtteranceStatus

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interviews (
    id          TEXT PRIMARY KEY,
    company     TEXT,
    stage       TEXT,
    created_at  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','active','completed'))
);

CREATE TABLE IF NOT EXISTS utterances (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    interview_id  TEXT NOT NULL REFERENCES interviews(id),
    speaker       TEXT NOT NULL CHECK (speaker IN ('interviewer','candidate')),
    start_ms      INTEGER NOT NULL,
    end_ms        INTEGER NOT NULL,
    audio_path    TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'queued'
                  CHECK (status IN ('queued','processing','completed','failed')),
    confidence    REAL,
    transcript    TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_utterances_interview ON utterances(interview_id);
CREATE INDEX IF NOT EXISTS idx_utterances_status ON utterances(status, id);
"""


class Repository:
    """Async data access for interviews and utterances.

    Uses WAL mode for concurrent reads/writes across threads.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = str(db_path)

    @property
    def db_path(self) -> str:
        return self._db_path

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

        # Verify tables exist
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] async for row in cursor]
            log.info("Database initialised at %s — tables: %s", self._db_path, tables)

    # ------------------------------------------------------------------
    # interview lifecycle
    # ------------------------------------------------------------------

    async def create_interview(
        self,
        interview_id: str,
        company: str | None = None,
        stage: str | None = None,
    ) -> Interview:
        interview = Interview(
            id=interview_id,
            company=company,
            stage=stage,
            created_at=datetime.now(tz=timezone.utc),
        )
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO interviews (id, company, stage, created_at, status) "
                "VALUES (?, ?, ?, ?, ?)",
                (interview.id, interview.company, interview.stage,
                 interview.created_at.isoformat(), interview.status.value),
            )
            await db.commit()
        log.info("Interview created: %s (company=%s, stage=%s)", interview_id, company, stage)
        return interview

    async def start_interview(self, interview_id: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE interviews SET status = ? WHERE id = ?",
                (InterviewStatus.ACTIVE.value, interview_id),
            )
            await db.commit()
        log.info("Interview started: %s", interview_id)

    async def finish_interview(self, interview_id: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE interviews SET status = ? WHERE id = ?",
                (InterviewStatus.COMPLETED.value, interview_id),
            )
            await db.commit()
        log.info("Interview finished: %s", interview_id)

    # ------------------------------------------------------------------
    # utterances
    # ------------------------------------------------------------------

    async def next_queued(self) -> Utterance | None:
        """Return the first queued utterance, or None."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM utterances WHERE status = 'queued' ORDER BY id LIMIT 1"
            )
            row = await cursor.fetchone()
            return self._row_to_utterance(row) if row else None

    async def count_pending(self, interview_id: str) -> int:
        """Count utterances still queued or processing for an interview."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM utterances WHERE interview_id = ? "
                "AND status IN ('queued','processing')",
                (interview_id,),
            )
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def mark_processing(self, utterance_id: int) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE utterances SET status = ? WHERE id = ?",
                (UtteranceStatus.PROCESSING.value, utterance_id),
            )
            await db.commit()
        log.debug("Utterance %d → processing", utterance_id)

    async def mark_completed(self, utterance_id: int, transcript: str, confidence: float) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE utterances SET status = ?, transcript = ?, confidence = ? WHERE id = ?",
                (UtteranceStatus.COMPLETED.value, transcript, confidence, utterance_id),
            )
            await db.commit()
        log.info("Utterance %d → completed (confidence=%.3f, text=%s)",
                 utterance_id, confidence, transcript[:80])

    async def mark_failed(self, utterance_id: int) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE utterances SET status = ? WHERE id = ?",
                (UtteranceStatus.FAILED.value, utterance_id),
            )
            await db.commit()
        log.warning("Utterance %d → failed", utterance_id)

    async def get_utterances(self, interview_id: str) -> list[Utterance]:
        """Return completed utterances for an interview, ordered by start_ms."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM utterances WHERE interview_id = ? AND status = 'completed' "
                "ORDER BY start_ms",
                (interview_id,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_utterance(r) for r in rows]

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------

    async def export_json(
        self,
        interview_id: str,
        model: str,
        language: str = "en",
    ) -> dict:
        """Return the final JSON payload matching the spec schema."""
        interview = await self._get_interview(interview_id)
        utterances = await self.get_utterances(interview_id)
        log.info("Exporting JSON for %s: %d utterances", interview_id, len(utterances))

        return {
            "schemaVersion": 1,
            "interviewId": interview_id,
            "company": interview.company if interview else None,
            "stage": interview.stage if interview else None,
            "transcriber": {
                "model": model,
                "language": language,
                "createdAt": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "utterances": [
                {
                    "id": u.id,
                    "speaker": u.speaker,
                    "startMs": u.start_ms,
                    "endMs": u.end_ms,
                    "confidence": u.confidence,
                    "text": u.transcript,
                }
                for u in utterances
                if u.transcript
            ],
        }

    async def export_txt(self, interview_id: str) -> str:
        """Return the transcript as a plain-text document."""
        interview = await self._get_interview(interview_id)
        utterances = await self.get_utterances(interview_id)

        lines: list[str] = []
        if interview:
            lines.append(f"Interview: {interview_id}")
            if interview.company:
                lines.append(f"Company: {interview.company}")
            if interview.stage:
                lines.append(f"Stage: {interview.stage}")
            lines.append(f"Date: {interview.created_at.strftime('%Y-%m-%d %H:%M UTC')}")
            lines.append("")

        for u in utterances:
            if u.transcript:
                ts = self._format_timestamp(u.start_ms / 1000)
                lines.append(f"[{ts}] {u.speaker}: {u.transcript}")

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

    async def _get_interview(self, interview_id: str) -> Interview | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM interviews WHERE id = ?", (interview_id,)
            )
            row = await cursor.fetchone()
            return self._row_to_interview(row) if row else None

    @staticmethod
    def _row_to_interview(row: aiosqlite.Row) -> Interview:
        return Interview(
            id=row["id"],
            company=row["company"],
            stage=row["stage"],
            created_at=datetime.fromisoformat(row["created_at"]),
            status=InterviewStatus(row["status"]),
        )

    @staticmethod
    def _row_to_utterance(row: aiosqlite.Row) -> Utterance:
        def _dt(val: str | None) -> datetime | None:
            return datetime.fromisoformat(val) if val else None

        return Utterance(
            id=row["id"],
            interview_id=row["interview_id"],
            speaker=row["speaker"],
            start_ms=row["start_ms"],
            end_ms=row["end_ms"],
            audio_path=row["audio_path"],
            status=UtteranceStatus(row["status"]),
            confidence=row["confidence"],
            transcript=row["transcript"],
            created_at=_dt(row["created_at"]),
        )
