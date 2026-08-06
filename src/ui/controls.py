"""Control bar with meeting action buttons."""

from __future__ import annotations

from enum import Enum, auto

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget


class ControlBar(QWidget):
    """Horizontal button bar: Start / Finish / Download / Upload."""

    class State(Enum):
        IDLE = auto()
        ACTIVE = auto()

    start_requested = Signal()
    finish_requested = Signal()
    download_requested = Signal()
    upload_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._start_btn = QPushButton("Start")
        self._finish_btn = QPushButton("Finish")
        self._download_btn = QPushButton("Download JSON")
        self._upload_btn = QPushButton("Upload")

        layout.addWidget(self._start_btn)
        layout.addWidget(self._finish_btn)
        layout.addWidget(self._download_btn)
        layout.addWidget(self._upload_btn)
        layout.addStretch()

        self._start_btn.clicked.connect(self.start_requested.emit)
        self._finish_btn.clicked.connect(self.finish_requested.emit)
        self._download_btn.clicked.connect(self.download_requested.emit)
        self._upload_btn.clicked.connect(self.upload_requested.emit)

        self.set_state(self.State.IDLE)
        self._download_btn.setEnabled(False)
        self._upload_btn.setEnabled(False)

    def set_state(self, state: State) -> None:
        self._start_btn.setEnabled(state == self.State.IDLE)
        self._finish_btn.setEnabled(state == self.State.ACTIVE)

    def set_download_enabled(self, enabled: bool) -> None:
        self._download_btn.setEnabled(enabled)

    def set_upload_enabled(self, enabled: bool) -> None:
        self._upload_btn.setEnabled(enabled)
