"""Control bar with meeting action buttons."""

from __future__ import annotations

from enum import Enum, auto

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget


class ControlBar(QWidget):
    """Horizontal button bar: Start / Pause / Resume / Finish / Settings."""

    class State(Enum):
        IDLE = auto()       # no meeting running
        ACTIVE = auto()     # capturing & transcribing
        PAUSED = auto()     # paused mid-meeting
        FINISHED = auto()   # just finished, resetting

    start_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    finish_requested = Signal()
    settings_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._start_btn = QPushButton("▶ Start")
        self._pause_btn = QPushButton("⏸ Pause")
        self._resume_btn = QPushButton("▶ Resume")
        self._finish_btn = QPushButton("⏹ Finish")
        self._settings_btn = QPushButton("⚙ Settings")

        layout.addWidget(self._start_btn)
        layout.addWidget(self._pause_btn)
        layout.addWidget(self._resume_btn)
        layout.addWidget(self._finish_btn)
        layout.addStretch()
        layout.addWidget(self._settings_btn)

        # connect
        self._start_btn.clicked.connect(self.start_requested.emit)
        self._pause_btn.clicked.connect(self.pause_requested.emit)
        self._resume_btn.clicked.connect(self.resume_requested.emit)
        self._finish_btn.clicked.connect(self.finish_requested.emit)
        self._settings_btn.clicked.connect(self.settings_requested.emit)

        self.set_state(self.State.IDLE)

    def set_state(self, state: State) -> None:
        """Enable / disable buttons based on meeting state."""
        self._start_btn.setEnabled(state == self.State.IDLE)
        self._pause_btn.setEnabled(state == self.State.ACTIVE)
        self._resume_btn.setEnabled(state == self.State.PAUSED)
        self._finish_btn.setEnabled(state in (self.State.ACTIVE, self.State.PAUSED))
        self._settings_btn.setEnabled(state == self.State.IDLE)
