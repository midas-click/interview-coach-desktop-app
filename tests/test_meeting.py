"""Tests for meeting controller lifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import Settings
from src.meeting.controller import MeetingController
from src.storage.models import InterviewStatus
from src.storage.repository import Repository


def _make_controller(tmp_path: Path) -> tuple[MeetingController, Repository, MagicMock, MagicMock]:
    db_path = tmp_path / "test.db"
    repo = Repository(db_path)

    async def _init():
        await repo.init()
    asyncio.run(_init())

    settings = Settings(
        whisper_model="base",
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / "temp",
    )

    with patch("src.meeting.controller.AudioManager") as mock_audio_cls, \
         patch("src.meeting.controller.TranscriptionEngine") as mock_eng_cls, \
         patch("src.meeting.controller.TranscriptionWorker") as mock_worker_cls:

        mock_audio = mock_audio_cls.return_value
        mock_worker = mock_worker_cls.return_value

        ctrl = MeetingController(settings, repo)
        return ctrl, repo, mock_audio, mock_worker


def test_create_interview(tmp_path: Path) -> None:
    ctrl, repo, _, _ = _make_controller(tmp_path)

    async def _test():
        mid = await ctrl.create_interview("Google", "System Design")
        assert len(mid) == 36  # full UUID
        interview = await repo._get_interview(mid)
        assert interview is not None
        assert interview.company == "Google"
        assert interview.stage == "System Design"
        assert interview.status == InterviewStatus.PENDING

    asyncio.run(_test())


def test_full_lifecycle(tmp_path: Path) -> None:
    ctrl, repo, mock_audio, mock_worker = _make_controller(tmp_path)

    async def _test():
        mid = await ctrl.create_interview("Meta", "Coding")

        # Mock finish: no pending utterances
        async def count_pending(interview_id):
            return 0
        repo.count_pending = count_pending

        await ctrl.start(mic_device=1)
        mock_audio.start.assert_called_once_with(mid, 1, None)
        mock_worker.start.assert_called_once()

        interview = await repo._get_interview(mid)
        assert interview is not None and interview.status == InterviewStatus.ACTIVE

        export = await ctrl.finish()
        mock_audio.stop_all.assert_called_once()
        mock_worker.stop.assert_called_once()

        assert export["interviewId"] == mid
        assert export["schemaVersion"] == 1
        assert export["company"] == "Meta"

    asyncio.run(_test())


def test_start_without_interview_raises(tmp_path: Path) -> None:
    ctrl, _, _, _ = _make_controller(tmp_path)

    async def _test():
        with pytest.raises(RuntimeError, match="No interview created"):
            await ctrl.start()

    asyncio.run(_test())


def test_export_json_format(tmp_path: Path) -> None:
    ctrl, repo, _, _ = _make_controller(tmp_path)

    async def _test():
        mid = await ctrl.create_interview("Stripe", "Phone Screen")

        # Insert a completed utterance directly
        import sqlite3
        conn = sqlite3.connect(repo.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "INSERT INTO utterances (interview_id, speaker, start_ms, end_ms, audio_path, status, confidence, transcript) "
            "VALUES (?, 'interviewer', 1250, 21450, '/tmp/x.wav', 'completed', 0.98, 'Tell me about yourself.')",
            (mid,),
        )
        conn.execute(
            "INSERT INTO utterances (interview_id, speaker, start_ms, end_ms, audio_path, status, confidence, transcript) "
            "VALUES (?, 'candidate', 21900, 61500, '/tmp/y.wav', 'completed', 0.96, 'Certainly. I have five years of experience.')",
            (mid,),
        )
        conn.commit()
        conn.close()

        export = await ctrl.export_current()
        assert export["schemaVersion"] == 1
        assert export["interviewId"] == mid
        assert export["company"] == "Stripe"
        assert export["stage"] == "Phone Screen"
        assert "transcriber" in export
        assert export["transcriber"]["model"] == "whisper-base"
        assert export["transcriber"]["language"] == "en"
        assert "createdAt" in export["transcriber"]
        assert len(export["utterances"]) == 2

        u0 = export["utterances"][0]
        assert u0["speaker"] == "interviewer"
        assert u0["startMs"] == 1250
        assert u0["endMs"] == 21450
        assert u0["confidence"] == 0.98
        assert u0["text"] == "Tell me about yourself."

        u1 = export["utterances"][1]
        assert u1["speaker"] == "candidate"
        assert u1["startMs"] == 21900
        assert u1["endMs"] == 61500
        assert u1["confidence"] == 0.96

    asyncio.run(_test())
