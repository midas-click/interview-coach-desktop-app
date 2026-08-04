"""Main application window.

Assembles the dashboard, controls, and transcript view into a single
window.  Wires MeetingController callbacks to UI updates.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from src.logger.logger import get_logger
from src.ui.controls import ControlBar
from src.ui.dashboard import Dashboard
from src.ui.transcript_view import TranscriptView

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

        self.setWindowTitle("Interview Transcriber")
        self.setMinimumSize(600, 420)

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
        self._controller.on_status_change = self._dashboard.set_status
        self._controller.on_chunks_persisted = self._on_chunks_persisted

        # elapsed timer
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_seconds = 0

        # restore persisted device selections
        self._restore_device_state()

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
                f"Found an unfinished meeting ({mid}). Finalise it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                await self._controller.finish()

    async def _on_start(self) -> None:
        self._transcript.clear()
        self._controls.set_word_count(0)
        self._controls.set_upload_enabled(False)
        self._controls.set_download_enabled(False)
        self._dashboard.set_upload_status("—")
        self._save_device_state()
        try:
            log.info("Start button clicked")
            company = self._dashboard.company_name.text().strip() or None
            stage = self._dashboard.interview_stage.text().strip() or None

            mic_dev = self._dashboard.selected_mic_device
            sys_dev = self._dashboard.selected_sys_device

            await self._controller.create_meeting(company, stage)
            log.info("Meeting created: %s", self._controller.meeting_id)
            await self._controller.start(mic_device=mic_dev, sys_device=sys_dev)

            self._dashboard.set_meeting_id(self._controller.meeting_id or "")
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
            self._dashboard.set_status("Finalising…")
            await self._controller.finish()
        except Exception:
            log.exception("Finish failed")

        self._dashboard.set_status("Idle")
        self._dashboard.set_upload_status("Ready to upload")
        self._dashboard.reset_inputs()
        self._controls.set_state(ControlBar.State.IDLE)
        self._controls.set_upload_enabled(True)
        self._controls.set_download_enabled(True)

    async def _on_download(self) -> None:
        try:
            export = await self._controller.export_current()
            mid = self._controller.meeting_id or "transcript"
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Transcript", f"{mid}.json",
                "JSON Files (*.json)",
            )
            if path:
                Path(path).write_text(json.dumps(export, indent=2), encoding="utf-8")
                self._dashboard.set_upload_status("Downloaded ✓")
        except Exception as exc:
            log.exception("Download failed")
            self._dashboard.set_upload_status(f"Failed: {exc}")

    async def _on_upload(self) -> None:
        try:
            self._dashboard.set_upload_status("Uploading…")
            export = await self._controller.export_current()
            await self._uploader.upload(export, self._controller.meeting_id or "unknown")
            self._dashboard.set_upload_status("Uploaded ✓")
        except Exception as exc:
            log.exception("Upload failed")
            self._dashboard.set_upload_status(f"Failed: {exc}")

    def _on_chunks_persisted(self, chunks) -> None:
        self._transcript.append_chunks(chunks)
        self._controls.set_word_count(self._transcript.word_count)

    def _start_elapsed(self) -> None:
        self._elapsed_seconds = 0
        self._controls.set_elapsed(0)
        self._elapsed_timer.start()

    def _tick_elapsed(self) -> None:
        self._elapsed_seconds += 1
        self._controls.set_elapsed(self._elapsed_seconds)

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
