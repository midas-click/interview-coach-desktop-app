"""Status dashboard showing meeting info, device selection, and status."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QWidget,
)

from src.audio.capture import list_microphones, list_loopback_devices


def _label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setTextFormat(Qt.TextFormat.PlainText)
    return lbl


def _value_label() -> QLabel:
    lbl = QLabel("—")
    lbl.setTextFormat(Qt.TextFormat.PlainText)
    lbl.setStyleSheet("font-weight: bold;")
    return lbl


class Dashboard(QFrame):
    """Top-of-window panel with meeting metadata and live status."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        grid = QGridLayout(self)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(4)

        # row 0 — meeting ID
        grid.addWidget(_label("Meeting:"), 0, 0)
        self.meeting_id_lbl = _value_label()
        grid.addWidget(self.meeting_id_lbl, 0, 1)

        # row 1 — company / stage inputs
        grid.addWidget(_label("Company:"), 1, 0)
        self.company_name = QLineEdit()
        self.company_name.setPlaceholderText("e.g. Google")
        self.company_name.setMaximumWidth(200)
        grid.addWidget(self.company_name, 1, 1)

        grid.addWidget(_label("Stage:"), 1, 2)
        self.interview_stage = QLineEdit()
        self.interview_stage.setPlaceholderText("e.g. Coding Interview")
        self.interview_stage.setMaximumWidth(200)
        grid.addWidget(self.interview_stage, 1, 3)

        # row 2 — mic / sys device selection
        grid.addWidget(_label("Microphone:"), 2, 0)
        self.mic_combo = QComboBox()
        self.mic_combo.addItem("(system default)", None)
        self._populate_mic_devices()
        self.mic_combo.setMaximumWidth(200)
        grid.addWidget(self.mic_combo, 2, 1)

        grid.addWidget(_label("System audio:"), 2, 2)
        self.sys_combo = QComboBox()
        self.sys_combo.addItem("(none — mic only)", None)
        self._populate_sys_devices()
        self.sys_combo.setMaximumWidth(200)
        grid.addWidget(self.sys_combo, 2, 3)

        # row 3 — (removed: status now in control bar)

    # -- public ----------------------------------------------------------

    @property
    def selected_mic_device(self) -> int | None:
        return self.mic_combo.currentData()

    @property
    def selected_sys_device(self) -> int | None:
        return self.sys_combo.currentData()

    def set_meeting_id(self, mid: str) -> None:
        self.meeting_id_lbl.setText(mid)

    def set_company_info(self, company: str | None, stage: str | None) -> None:
        self.company_name.setReadOnly(True)
        self.interview_stage.setReadOnly(True)
        self.mic_combo.setEnabled(False)
        self.sys_combo.setEnabled(False)

    def reset_inputs(self) -> None:
        """Re-enable inputs for a new meeting."""
        self.company_name.setReadOnly(False)
        self.interview_stage.setReadOnly(False)
        self.company_name.clear()
        self.interview_stage.clear()
        self.mic_combo.setEnabled(True)
        self.sys_combo.setEnabled(True)
        self.meeting_id_lbl.setText("—")

    # -- device population ----------------------------------------------

    def _populate_mic_devices(self) -> None:
        for d in list_microphones():
            self.mic_combo.addItem(d["name"], d["index"])

    def _populate_sys_devices(self) -> None:
        for d in list_loopback_devices():
            self.sys_combo.addItem(d["name"], d["index"])
