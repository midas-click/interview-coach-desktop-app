"""Interview Transcriber — Windows desktop application for real-time transcription.

Entry point. Initialises Qt application with asyncio event loop,
loads configuration, and launches the main window.
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from pathlib import Path

import qasync
from PySide6.QtWidgets import QApplication

from src.config.settings import Settings
from src.logger.logger import get_logger, setup_logging
from src.meeting.controller import MeetingController
from src.storage.repository import Repository
from src.upload.s3_uploader import S3Uploader
from src.ui.main_window import MainWindow

# ── Crash logging (catches silent C-level crashes from ctranslate2 / av) ──

_CRASH_DIR = Path(sys.executable).parent / "_internal" / "logs" if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent / "logs"
_CRASH_LOG = _CRASH_DIR / "crash.log"


def _write_crash(exc_type, exc_value, _tb) -> None:
    _CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_CRASH_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"CRASH: {exc_type.__name__ if exc_type else 'unknown'}: {exc_value}\n")
        traceback.print_exception(exc_type, exc_value, _tb, file=f)


def _install_crash_handler() -> None:
    sys.excepthook = _write_crash
    # Catch segfaults / illegal instructions from C extensions
    try:
        import signal
        import ctypes

        def _on_signal(sig, _frame):
            sig_names = {
                signal.SIGSEGV: "SIGSEGV (segfault)",
                signal.SIGABRT: "SIGABRT (abort)",
                signal.SIGILL: "SIGILL (illegal instruction)",
                signal.SIGFPE: "SIGFPE (float error)",
            }
            name = sig_names.get(sig, f"signal {sig}")
            _CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(_CRASH_LOG, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"CRASH: {name}\n")
                f.write(f"This usually means a C extension (ctranslate2, av) is incompatible with this CPU.\n")
            sys.exit(1)

        signal.signal(signal.SIGSEGV, _on_signal)
        signal.signal(signal.SIGABRT, _on_signal)
        signal.signal(signal.SIGILL, _on_signal)
        signal.signal(signal.SIGFPE, _on_signal)
    except Exception:
        pass  # signal handlers best-effort


_install_crash_handler()

# ────────────────────────────────────────────────────────────────────────────


def _app_dir() -> Path:
    """Directory containing the app bundle (or source tree in dev)."""
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
        internal = base / "_internal"
        return internal if internal.exists() else base
    return Path(__file__).resolve().parent.parent


DB_PATH = _app_dir() / "data" / "transcriber.db"


def main() -> None:
    # Windows taskbar icon — must be set before QApplication is shown.
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "notepadder.app"
        )
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName("Notepadder")
    app.setOrganizationName("Notepadder")
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
    # Set the app-level icon too so the taskbar picks it up.
    if not window.windowIcon().isNull():
        app.setWindowIcon(window.windowIcon())
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
