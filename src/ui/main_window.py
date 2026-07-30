"""Main application window.

Assembles the dashboard, controls, and transcript view into a single
window.  Wires MeetingController callbacks to UI updates.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.ui.controls import ControlBar
from src.ui.dashboard import Dashboard
from src.ui.transcript_view import TranscriptView

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

        self.setWindowTitle("Interview Transcriber")
        self.setMinimumSize(900, 600)

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

        # transcript (takes remaining space)
        self._transcript = TranscriptView()
        layout.addWidget(self._transcript, stretch=1)

        # -- wire signals ---------------------------------------------------
        self._controls.start_requested.connect(self._on_start)
        self._controls.pause_requested.connect(self._on_pause)
        self._controls.resume_requested.connect(self._on_resume)
        self._controls.finish_requested.connect(self._on_finish)
        self._controls.settings_requested.connect(self._on_settings)

        # controller callbacks → UI
        self._controller.on_status_change = self._dashboard.set_status
        self._controller.on_chunks_persisted = self._transcript.append_chunks

        # elapsed timer
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_seconds = 0

        # initial state
        self._controls.set_state(ControlBar.State.IDLE)

        # check for crash recovery
        asyncio.ensure_future(self._check_recovery())

    # ------------------------------------------------------------------
    # slots
    # ------------------------------------------------------------------

    async def _check_recovery(self) -> None:
        mid = await self._controller.recover_active_meeting()
        if mid is not None:
            reply = QMessageBox.question(
                self,
                "Recover Meeting",
                f"Found an unfinished meeting ({mid}). Resume it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                await self._controller.resume()
                self._start_elapsed()
                self._controls.set_state(ControlBar.State.ACTIVE)
            else:
                await self._controller.finish()

    async def _on_start(self) -> None:
        company = self._dashboard.company_name.text().strip() or None
        stage = self._dashboard.interview_stage.text().strip() or None

        mic_dev = self._settings.microphone_device
        sys_dev = self._settings.system_audio_device

        await self._controller.create_meeting(company, stage)
        await self._controller.start(mic_device=mic_dev, sys_device=sys_dev)

        self._dashboard.set_meeting_id(self._controller.meeting_id or "")
        self._dashboard.set_company_info(company, stage)
        self._start_elapsed()
        self._controls.set_state(ControlBar.State.ACTIVE)

    async def _on_pause(self) -> None:
        await self._controller.pause()
        self._elapsed_timer.stop()
        self._controls.set_state(ControlBar.State.PAUSED)

    async def _on_resume(self) -> None:
        mic_dev = self._settings.microphone_device
        sys_dev = self._settings.system_audio_device
        await self._controller.resume(mic_device=mic_dev, sys_device=sys_dev)
        self._start_elapsed()
        self._controls.set_state(ControlBar.State.ACTIVE)

    async def _on_finish(self) -> None:
        self._elapsed_timer.stop()
        self._controls.set_state(ControlBar.State.FINISHED)

        try:
            export = await self._controller.finish()
            self._dashboard.set_status("Exporting…")

            if self._settings.auto_upload and self._settings.aws_bucket:
                await self._uploader.upload(export, self._controller.meeting_id or "unknown")
                self._dashboard.set_upload_status("Uploaded ✓")
            else:
                self._dashboard.set_upload_status("Saved locally")
        except Exception as exc:
            self._dashboard.set_upload_status(f"Upload failed: {exc}")

        self._controls.set_state(ControlBar.State.IDLE)

    def _on_settings(self) -> None:
        from src.ui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(self._settings, self)
        if dialog.exec():
            from src.config.settings import save_settings
            save_settings(self._settings, dialog.config_path)

    def _start_elapsed(self) -> None:
        self._elapsed_seconds = 0
        self._dashboard.set_elapsed(0)
        self._elapsed_timer.start()

    def _tick_elapsed(self) -> None:
        self._elapsed_seconds += 1
        self._dashboard.set_elapsed(self._elapsed_seconds)
