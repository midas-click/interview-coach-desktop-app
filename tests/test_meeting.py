"""Tests for meeting controller lifecycle."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import Settings
from src.meeting.controller import MeetingController
from src.storage.models import MeetingStatus
from src.storage.repository import Repository
from src.transcription.engine import TranscriptionSegment


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_controller(tmp_path: Path) -> tuple[MeetingController, Repository, MagicMock, MagicMock]:
    """Create a controller wired to a real Repository with mocked audio/engine."""
    db_path = tmp_path / "test.db"
    repo = Repository(db_path)

    async def _init():
        await repo.init()

    asyncio.run(_init())

    settings = Settings(whisper_model="base")

    # mock AudioManager so no real audio hardware is needed
    with patch("src.meeting.controller.AudioManager") as mock_audio_cls, \
         patch("src.meeting.controller.TranscriptionEngine") as mock_eng_cls:

        mock_audio = mock_audio_cls.return_value
        mock_audio.microphone_state.name = "STOPPED"
        mock_audio.system_audio_state.name = "STOPPED"
        mock_audio.is_any_running = False

        mock_engine = mock_eng_cls.return_value
        mock_engine.stop.return_value = []

        ctrl = MeetingController(settings, repo)
        return ctrl, repo, mock_audio, mock_engine


def _segment(text: str, start: float, end: float) -> TranscriptionSegment:
    return TranscriptionSegment(text=text, start=start, end=end, confidence=0.9)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_create_meeting(tmp_path: Path) -> None:
    ctrl, repo, _, _ = _make_controller(tmp_path)

    async def _test():
        mid = await ctrl.create_meeting("Google", "System Design")
        assert len(mid) == 12
        meeting = await repo.get_meeting(mid)
        assert meeting is not None
        assert meeting.company_name == "Google"
        assert meeting.interview_stage == "System Design"

    asyncio.run(_test())


def test_full_lifecycle(tmp_path: Path) -> None:
    ctrl, repo, mock_audio, mock_engine = _make_controller(tmp_path)

    # capture the transcription callback when engine starts
    engine_started = threading.Event()

    def fake_start(audio_queue, on_segment):
        mock_engine._on_segment = on_segment
        engine_started.set()

    mock_engine.start.side_effect = fake_start

    async def _test():
        await ctrl.create_meeting("Meta", "Coding")
        mid = ctrl.meeting_id

        # start
        await ctrl.start(mic_device=1)
        assert engine_started.wait(2)

        meeting = await repo.get_meeting(mid)
        assert meeting is not None and meeting.status == MeetingStatus.ACTIVE

        # simulate transcription segments
        ctrl._on_transcription_segment(_segment("Tell me about yourself.", 1.0, 3.0))
        ctrl._on_transcription_segment(_segment("I have 10 years of experience.", 4.0, 7.0))

        # finish
        export = await ctrl.finish()
        assert export["meetingId"] == mid
        assert export["companyName"] == "Meta"
        assert len(export["transcript"]) == 2

        meeting = await repo.get_meeting(mid)
        assert meeting is not None and meeting.status == MeetingStatus.FINISHED

    asyncio.run(_test())


def test_start_without_meeting_raises(tmp_path: Path) -> None:
    ctrl, _, _, _ = _make_controller(tmp_path)

    async def _test():
        with pytest.raises(RuntimeError, match="No meeting created"):
            await ctrl.start()

    asyncio.run(_test())


def test_recover_active_meeting(tmp_path: Path) -> None:
    ctrl, repo, _, _ = _make_controller(tmp_path)

    async def _test():
        await repo.create_meeting("recover-123")
        await repo.start_meeting("recover-123")

        mid = await ctrl.recover_active_meeting()
        assert mid == "recover-123"
        assert ctrl.meeting_id == "recover-123"

    asyncio.run(_test())


def test_recover_no_active_meeting(tmp_path: Path) -> None:
    ctrl, _, _, _ = _make_controller(tmp_path)

    async def _test():
        mid = await ctrl.recover_active_meeting()
        assert mid is None

    asyncio.run(_test())
