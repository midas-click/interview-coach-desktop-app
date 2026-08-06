"""Status dashboard showing interview metadata, device selection, and live status."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from src.audio.capture import list_microphones, list_loopback_devices

_FIELD_WIDTH = 240


def _row(label_text: str, widget: QWidget) -> QHBoxLayout:
    lbl = QLabel(label_text)
    lbl.setTextFormat(Qt.TextFormat.PlainText)
    lbl.setFixedWidth(100)
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(lbl)
    row.addWidget(widget)
    row.addStretch()
    return row


class Dashboard(QFrame):
    """Top-of-window panel with interview metadata, device selection, and status."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # -- meeting ID -----------------------------------------------------
        self.meeting_id_lbl = QLabel("—")
        self.meeting_id_lbl.setTextFormat(Qt.TextFormat.PlainText)
        layout.addLayout(_row("Meeting:", self.meeting_id_lbl))

        # -- company --------------------------------------------------------
        self.company_name = QLineEdit()
        self.company_name.setPlaceholderText("e.g. Google")
        self.company_name.setFixedWidth(_FIELD_WIDTH)
        layout.addLayout(_row("Company:", self.company_name))

        # -- stage ----------------------------------------------------------
        self.interview_stage = QLineEdit()
        self.interview_stage.setPlaceholderText("e.g. Coding Interview")
        self.interview_stage.setFixedWidth(_FIELD_WIDTH)
        layout.addLayout(_row("Stage:", self.interview_stage))

        # -- microphone -----------------------------------------------------
        self.mic_combo = QComboBox()
        self.mic_combo.addItem("(system default)", None)
        self._populate_mic_devices()
        self.mic_combo.setFixedWidth(_FIELD_WIDTH)
        layout.addLayout(_row("Microphone:", self.mic_combo))

        # -- system audio ---------------------------------------------------
        self.sys_combo = QComboBox()
        self.sys_combo.addItem("(none — mic only)", None)
        self._populate_sys_devices()
        self.sys_combo.setFixedWidth(_FIELD_WIDTH)
        layout.addLayout(_row("System audio:", self.sys_combo))

        # -- status ---------------------------------------------------------
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setTextFormat(Qt.TextFormat.PlainText)
        layout.addLayout(_row("Status:", self.status_lbl))

        # -- elapsed --------------------------------------------------------
        self.elapsed_lbl = QLabel("00:00")
        self.elapsed_lbl.setTextFormat(Qt.TextFormat.PlainText)
        layout.addLayout(_row("Elapsed:", self.elapsed_lbl))

    # -- public ----------------------------------------------------------

    @property
    def selected_mic_device(self) -> int | None:
        return self.mic_combo.currentData()

    @property
    def selected_sys_device(self) -> int | None:
        return self.sys_combo.currentData()

    def set_interview_id(self, mid: str) -> None:
        self.meeting_id_lbl.setText(mid)

    def set_status(self, text: str) -> None:
        self.status_lbl.setText(text)

    def set_elapsed(self, seconds: int) -> None:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h:
            self.elapsed_lbl.setText(f"{h}:{m:02d}:{s:02d}")
        else:
            self.elapsed_lbl.setText(f"{m:02d}:{s:02d}")

    def set_company_info(self, company: str | None, stage: str | None) -> None:
        self.company_name.setReadOnly(True)
        self.interview_stage.setReadOnly(True)
        self.mic_combo.setEnabled(False)
        self.sys_combo.setEnabled(False)

    def reset_inputs(self) -> None:
        self.company_name.setReadOnly(False)
        self.interview_stage.setReadOnly(False)
        self.company_name.clear()
        self.interview_stage.clear()
        self.mic_combo.setEnabled(True)
        self.sys_combo.setEnabled(True)
        self.meeting_id_lbl.setText("—")
        self.status_lbl.setText("Ready")
        self.elapsed_lbl.setText("00:00")

    # -- device population ----------------------------------------------

    def _populate_mic_devices(self) -> None:
        for d in list_microphones():
            self.mic_combo.addItem(d["name"], d["index"])

    def _populate_sys_devices(self) -> None:
        for d in list_loopback_devices():
            self.sys_combo.addItem(d["name"], d["index"])
