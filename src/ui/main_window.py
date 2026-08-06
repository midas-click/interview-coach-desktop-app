"""Main application window.

Assembles the dashboard and controls into a single window.
Wires MeetingController callbacks to UI updates.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMenu,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from src.logger.logger import get_logger
from src.ui.controls import ControlBar
from src.ui.dashboard import Dashboard

log = get_logger(__name__)

DEVICE_STATE_PATH = Path("data/device_state.json")

if TYPE_CHECKING:
    from src.config.settings import Settings
    from src.meeting.controller import MeetingController
    from src.upload.s3_uploader import S3Uploader


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(
        self,
        settings: Settings,
        controller: MeetingController,
        uploader: S3Uploader,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._controller = controller
        self._uploader = uploader

        self.setWindowTitle("Notepadder")
        self.setMinimumSize(360, 300)
        self._load_icon()

        # -- central widget -------------------------------------------------
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # dashboard
        self._dashboard = Dashboard()
        layout.addWidget(self._dashboard)

        # controls
        self._controls = ControlBar()
        layout.addWidget(self._controls)

        # -- wire signals ---------------------------------------------------
        self._controls.start_requested.connect(
            lambda: asyncio.ensure_future(self._on_start())
        )
        self._controls.finish_requested.connect(
            lambda: asyncio.ensure_future(self._on_finish())
        )
        self._controls.download_requested.connect(
            lambda: asyncio.ensure_future(self._on_download())
        )
        self._controls.upload_requested.connect(
            lambda: asyncio.ensure_future(self._on_upload())
        )

        # controller callbacks → UI
        self._controller.on_status_change = self._set_status
        self._controller.audio.on_speaking_change = self._on_speaking_change

        # elapsed timer
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_seconds = 0

        # restore persisted device selections
        self._restore_device_state()

        # initial state
        self._set_status("Ready")
        self._controls.set_state(ControlBar.State.IDLE)

        # tray icon
        self._tray = QSystemTrayIcon(self)
        self._tray.activated.connect(self._on_tray_activated)
        tray_menu = QMenu()
        tray_menu.addAction("Show", self._show_from_tray)
        tray_menu.addAction("Quit", self._really_quit)
        self._tray.setContextMenu(tray_menu)
        if not self.windowIcon().isNull():
            self._tray.setIcon(self.windowIcon())
        self._tray.setToolTip("Notepadder")

    def _load_icon(self) -> None:
        if getattr(sys, "frozen", False):
            base = Path(sys.executable).parent
            internal = base / "_internal"
            root = internal if internal.exists() else base
        else:
            root = Path(__file__).resolve().parent.parent.parent
        icon_path = root / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    # ------------------------------------------------------------------
    # tray icon
    # ------------------------------------------------------------------

    def changeEvent(self, event) -> None:
        if event.type() == event.Type.WindowStateChange and self.isMinimized():
            self.hide()
            self._tray.show()
            event.ignore()
        else:
            super().changeEvent(event)

    def closeEvent(self, event) -> None:
        self._tray.hide()
        super().closeEvent(event)

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._show_from_tray()

    def _show_from_tray(self) -> None:
        self._tray.hide()
        self.showNormal()
        self.activateWindow()

    def _really_quit(self) -> None:
        self._tray.hide()
        from PySide6.QtWidgets import QApplication
        QApplication.instance().quit()

    # ------------------------------------------------------------------
    # slots
    # ------------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        self._dashboard.set_status(text)

    def _on_speaking_change(self, speaker: str | None) -> None:
        """Called from capture thread when the active speaker changes."""
        if speaker == "interviewer":
            self._set_status("Interviewer speaking…")
        elif speaker == "candidate":
            self._set_status("Candidate speaking…")
        else:
            self._set_status("Actively listening…")
        log.debug("Speaking change: %s", speaker)

    async def _on_start(self) -> None:
        self._controls.set_download_enabled(False)
        self._controls.set_upload_enabled(False)
        self._save_device_state()
        try:
            log.info("Start button clicked")
            company = self._dashboard.company_name.text().strip() or None
            stage = self._dashboard.interview_stage.text().strip() or None

            mic_dev = self._dashboard.selected_mic_device
            sys_dev = self._dashboard.selected_sys_device

            await self._controller.create_interview(company, stage)
            log.info("Interview created: %s", self._controller.interview_id)
            await self._controller.start(mic_device=mic_dev, sys_device=sys_dev)

            self._dashboard.set_interview_id(self._controller.interview_id or "")
            self._dashboard.set_company_info(company, stage)
            self._start_elapsed()
            self._controls.set_state(ControlBar.State.ACTIVE)
        except Exception:
            log.exception("Start failed")

    async def _on_finish(self) -> None:
        log.info("Finish button clicked")
        self._elapsed_timer.stop()
        self._controls.set_state(ControlBar.State.IDLE)

        try:
            self._set_status("Finalising…")
            await self._controller.finish()
        except Exception:
            log.exception("Finish failed")

        self._controls.set_state(ControlBar.State.IDLE)
        self._controls.set_download_enabled(True)
        self._controls.set_upload_enabled(True)
        self._dashboard.reset_inputs()

    async def _on_download(self) -> None:
        try:
            self._set_status("Downloading JSON…")
            export = await self._controller.export_current()
            mid = self._controller.interview_id or "transcript"
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Transcript", f"{mid}.json",
                "JSON Files (*.json)",
            )
            if path:
                Path(path).write_text(json.dumps(export, indent=2), encoding="utf-8")
                self._set_status("Download complete")
        except Exception as exc:
            log.exception("Download failed")
            self._set_status(f"Download failed: {exc}")

    async def _on_upload(self) -> None:
        try:
            self._set_status("Uploading…")
            export = await self._controller.export_current()
            await self._uploader.upload(export, self._controller.interview_id or "unknown")
            self._set_status("Upload complete")
        except Exception as exc:
            log.exception("Upload failed")
            self._set_status(f"Upload failed: {exc}")

    def _start_elapsed(self) -> None:
        self._elapsed_seconds = 0
        self._dashboard.set_elapsed(0)
        self._elapsed_timer.start()

    def _tick_elapsed(self) -> None:
        self._elapsed_seconds += 1
        self._dashboard.set_elapsed(self._elapsed_seconds)

    # -- device state persistence --------------------------------------

    def _save_device_state(self) -> None:
        DEVICE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEVICE_STATE_PATH.write_text(json.dumps({
            "microphone_device": self._dashboard.selected_mic_device,
            "system_audio_device": self._dashboard.selected_sys_device,
        }))

    def _restore_device_state(self) -> None:
        if not DEVICE_STATE_PATH.exists():
            return
        try:
            state = json.loads(DEVICE_STATE_PATH.read_text())
            mic_idx = state.get("microphone_device")
            sys_idx = state.get("system_audio_device")
            if mic_idx is not None:
                for i in range(self._dashboard.mic_combo.count()):
                    if self._dashboard.mic_combo.itemData(i) == mic_idx:
                        self._dashboard.mic_combo.setCurrentIndex(i)
                        break
            if sys_idx is not None:
                for i in range(self._dashboard.sys_combo.count()):
                    if self._dashboard.sys_combo.itemData(i) == sys_idx:
                        self._dashboard.sys_combo.setCurrentIndex(i)
                        break
        except Exception:
            pass
