"""Status dashboard showing meeting info, elapsed time, and indicators."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QWidget,
)


def _label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setTextFormat(Qt.TextFormat.PlainText)
    return lbl


def _value_label() -> QLabel:
    lbl = QLabel("—")
    lbl.setTextFormat(Qt.TextFormat.PlainText)
    lbl.setStyleSheet("font-weight: bold;")
    return lbl


class Dashboard(QWidget):
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

        # row 2 — elapsed
        grid.addWidget(_label("Elapsed:"), 2, 0)
        self.elapsed_lbl = _value_label()
        grid.addWidget(self.elapsed_lbl, 2, 1)

        # row 3 — status + upload
        grid.addWidget(_label("Status:"), 3, 0)
        self.status_lbl = QLabel("Idle")
        grid.addWidget(self.status_lbl, 3, 1)

        grid.addWidget(_label("Upload:"), 3, 2)
        self.upload_lbl = QLabel("—")
        grid.addWidget(self.upload_lbl, 3, 3)

        # row 4 — mic / sys indicators
        grid.addWidget(_label("Mic:"), 4, 0)
        self.mic_lbl = QLabel("○")
        grid.addWidget(self.mic_lbl, 4, 1)

        grid.addWidget(_label("System audio:"), 4, 2)
        self.sys_lbl = QLabel("○")
        grid.addWidget(self.sys_lbl, 4, 3)

    # -- public ----------------------------------------------------------

    def set_meeting_id(self, mid: str) -> None:
        self.meeting_id_lbl.setText(mid)

    def set_company_info(self, company: str | None, stage: str | None) -> None:
        self.company_name.setReadOnly(True)
        self.interview_stage.setReadOnly(True)

    def set_elapsed(self, seconds: int) -> None:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h:
            self.elapsed_lbl.setText(f"{h}:{m:02d}:{s:02d}")
        else:
            self.elapsed_lbl.setText(f"{m:02d}:{s:02d}")

    def set_status(self, status: str) -> None:
        self.status_lbl.setText(status)

    def set_upload_status(self, text: str) -> None:
        self.upload_lbl.setText(text)
