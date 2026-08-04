"""Control bar with meeting action buttons."""

from __future__ import annotations

from enum import Enum, auto

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


class ControlBar(QWidget):
    """Horizontal button bar: Start / Finish / Upload."""

    class State(Enum):
        IDLE = auto()       # no meeting running
        ACTIVE = auto()     # capturing & transcribing

    start_requested = Signal()
    finish_requested = Signal()
    upload_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._start_btn = QPushButton("Start")
        self._finish_btn = QPushButton("Finish")
        self._upload_btn = QPushButton("Upload")

        layout.addWidget(self._start_btn)
        layout.addWidget(self._finish_btn)
        layout.addWidget(self._upload_btn)
        layout.addStretch()

        self._words_lbl = QLabel("0 words")
        self._words_lbl.setStyleSheet("color: #888; font-size: 12px;")
        self._words_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._words_lbl)

        self._elapsed_lbl = QLabel("00:00")
        self._elapsed_lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
        self._elapsed_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._elapsed_lbl)

        # connect
        self._start_btn.clicked.connect(self.start_requested.emit)
        self._finish_btn.clicked.connect(self.finish_requested.emit)
        self._upload_btn.clicked.connect(self.upload_requested.emit)

        self.set_state(self.State.IDLE)
        self._upload_btn.setEnabled(False)

    def set_state(self, state: State) -> None:
        """Enable / disable buttons based on meeting state."""
        self._start_btn.setEnabled(state == self.State.IDLE)
        self._finish_btn.setEnabled(state == self.State.ACTIVE)

    def set_upload_enabled(self, enabled: bool) -> None:
        self._upload_btn.setEnabled(enabled)

    def set_elapsed(self, seconds: int) -> None:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h:
            self._elapsed_lbl.setText(f"{h}:{m:02d}:{s:02d}")
        else:
            self._elapsed_lbl.setText(f"{m:02d}:{s:02d}")

    def set_word_count(self, count: int) -> None:
        self._words_lbl.setText(f"{count} words")
