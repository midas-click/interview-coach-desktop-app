"""Interview Transcriber — Windows desktop application for real-time transcription.

Entry point. Initialises Qt application with asyncio event loop,
loads configuration, and launches the main window.
"""

import sys
import asyncio
from pathlib import Path

import qasync
from PySide6.QtWidgets import QApplication


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Interview Transcriber")
    app.setOrganizationName("InterviewCoach")
    app.setApplicationVersion("0.1.0")

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    # TODO: load config → initialise controller → show main window

    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
