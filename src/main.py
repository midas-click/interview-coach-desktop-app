"""Interview Transcriber — Windows desktop application for real-time transcription.

Entry point. Initialises Qt application with asyncio event loop,
loads configuration, and launches the main window.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import qasync
from PySide6.QtWidgets import QApplication

from src.config.settings import Settings
from src.logger.logger import get_logger, setup_logging
from src.meeting.controller import MeetingController
from src.storage.repository import Repository
from src.upload.s3_uploader import S3Uploader
from src.ui.main_window import MainWindow


def _app_dir() -> Path:
    """Directory containing the app bundle (or source tree in dev)."""
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
        internal = base / "_internal"
        return internal if internal.exists() else base
    return Path(__file__).resolve().parent.parent


DB_PATH = _app_dir() / "data" / "transcriber.db"


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Interview Transcriber")
    app.setOrganizationName("InterviewCoach")
    app.setApplicationVersion("0.1.0")

    # --- config & logging ------------------------------------------------
    settings = Settings()

    # When running as a frozen bundle, use paths relative to the app dir
    # and auto-detect the bundled whisper model.
    if getattr(sys, "frozen", False):
        base = _app_dir()
        settings.output_dir = base / "output"
        settings.log_file = base / "logs" / "transcriber.log"
        bundled_model = base / "models" / "whisper-base"
        if bundled_model.exists():
            settings.whisper_model = str(bundled_model)

    setup_logging(settings)
    log = get_logger(__name__)
    log.info("Application starting")

    # --- data layer ------------------------------------------------------
    repo = Repository(DB_PATH)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    async def _init() -> None:
        await repo.init()

    loop.run_until_complete(_init())

    # --- core modules ----------------------------------------------------
    controller = MeetingController(settings, repo)
    uploader = S3Uploader(settings)

    # --- UI --------------------------------------------------------------
    window = MainWindow(settings, controller, uploader)
    window.show()

    log.info("Main window shown")

    # --- run -------------------------------------------------------------
    app.aboutToQuit.connect(loop.stop)
    loop.run_forever()

    # Suppress known qasync shutdown error on Windows where Qt deletes
    # signal sources before the event loop fully closes.
    try:
        loop.close()
    except RuntimeError:
        pass


if __name__ == "__main__":
    main()
