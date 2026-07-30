"""Scrollable live-transcript display."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import QFrame, QTextEdit, QVBoxLayout, QWidget

from src.storage.models import TranscriptChunk


class TranscriptView(QWidget):
    """Read-only text area that appends transcript chunks as they arrive."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._editor = QTextEdit()
        self._editor.setReadOnly(True)
        font = QFont("Segoe UI", 11)
        self._editor.setFont(font)
        self._editor.setPlaceholderText("Transcription will appear here…")
        layout.addWidget(self._editor)

        self._word_count = 0

    def append_chunks(self, chunks: list[TranscriptChunk]) -> None:
        """Append new transcript chunks to the display."""
        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        for chunk in chunks:
            if chunk.text.strip():
                cursor.insertText(chunk.text.strip() + " ")
                self._word_count += len(chunk.text.split())

        # auto-scroll to bottom
        self._editor.setTextCursor(cursor)
        self._editor.ensureCursorVisible()

    @property
    def word_count(self) -> int:
        return self._word_count

    def clear(self) -> None:
        self._editor.clear()
        self._word_count = 0
