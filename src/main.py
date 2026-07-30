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

from src.config.settings import load_settings
from src.logger.logger import get_logger, setup_logging
from src.meeting.controller import MeetingController
from src.storage.repository import Repository
from src.upload.s3_uploader import S3Uploader
from src.ui.main_window import MainWindow

CONFIG_PATH = Path("config.yaml")
DB_PATH = Path("data/transcriber.db")


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Interview Transcriber")
    app.setOrganizationName("InterviewCoach")
    app.setApplicationVersion("0.1.0")

    # --- config & logging ------------------------------------------------
    settings = load_settings(CONFIG_PATH)
    setup_logging(settings)
    log = get_logger(__name__)
    log.info("Application starting")

    # --- data layer ------------------------------------------------------
    repo = Repository(DB_PATH)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    async def _init() -> None:
        await repo.init()

    with loop:
        loop.run_until_complete(_init())

    # --- core modules ----------------------------------------------------
    controller = MeetingController(settings, repo)
    uploader = S3Uploader(settings)

    # --- UI --------------------------------------------------------------
    window = MainWindow(settings, controller, uploader)
    window.show()

    log.info("Main window shown")
    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
