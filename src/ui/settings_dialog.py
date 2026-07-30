"""Settings dialog for editing all user-configurable options."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt as _Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.audio.capture import list_audio_devices
from src.config.settings import VALID_LOG_LEVELS, VALID_WHISPER_MODELS, Settings


class SettingsDialog(QDialog):
    """Modal dialog for editing application settings.

    On accept, the *settings* object is mutated in-place and the
    caller is expected to persist it via ``save_settings()``.
    """

    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)
        self._settings = settings
        self.config_path = Path("config.yaml")  # default; caller may override

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(_Qt.AlignmentFlag.AlignLeft | _Qt.AlignmentFlag.AlignVCenter)

        # --- whisper model -------------------------------------------------
        self._whisper_cb = QComboBox()
        self._whisper_cb.addItems(sorted(VALID_WHISPER_MODELS))
        self._whisper_cb.setCurrentText(settings.whisper_model)
        form.addRow("Whisper model:", self._whisper_cb)

        # --- audio devices -------------------------------------------------
        devices = list_audio_devices()
        device_choices = ["(default)"] + [
            f"{d['name']} [{d['index']}]" for d in devices
        ]

        self._mic_cb = QComboBox()
        self._mic_cb.addItems(device_choices)
        form.addRow("Microphone:", self._mic_cb)

        self._sys_cb = QComboBox()
        self._sys_cb.addItems(device_choices)
        form.addRow("System audio:", self._sys_cb)

        # pre-select current devices
        if settings.microphone_device is not None:
            idx = next(
                (i for i, d in enumerate(devices) if d["index"] == settings.microphone_device),
                None,
            )
            if idx is not None:
                self._mic_cb.setCurrentIndex(idx + 1)  # +1 for "(default)"

        if settings.system_audio_device is not None:
            idx = next(
                (i for i, d in enumerate(devices) if d["index"] == settings.system_audio_device),
                None,
            )
            if idx is not None:
                self._sys_cb.setCurrentIndex(idx + 1)

        # --- output dir ----------------------------------------------------
        out_row = QHBoxLayout()
        self._output_edit = QLineEdit(str(settings.output_dir))
        out_row.addWidget(self._output_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_output)
        out_row.addWidget(browse_btn)
        form.addRow("Output directory:", out_row)

        # --- auto upload ---------------------------------------------------
        self._auto_upload_cb = QCheckBox()
        self._auto_upload_cb.setChecked(settings.auto_upload)
        form.addRow("Auto-upload to S3:", self._auto_upload_cb)

        # --- AWS -----------------------------------------------------------
        self._aws_region = QLineEdit(settings.aws_region)
        form.addRow("AWS region:", self._aws_region)

        self._aws_bucket = QLineEdit(settings.aws_bucket)
        form.addRow("AWS bucket:", self._aws_bucket)

        self._aws_key = QLineEdit(settings.aws_access_key_id)
        self._aws_key.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("AWS access key:", self._aws_key)

        self._aws_secret = QLineEdit(settings.aws_secret_access_key)
        self._aws_secret.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("AWS secret key:", self._aws_secret)

        # --- logging -------------------------------------------------------
        self._log_level_cb = QComboBox()
        self._log_level_cb.addItems(sorted(VALID_LOG_LEVELS))
        self._log_level_cb.setCurrentText(settings.log_level)
        form.addRow("Log level:", self._log_level_cb)

        self._log_file = QLineEdit(str(settings.log_file))
        form.addRow("Log file:", self._log_file)

        layout.addLayout(form)

        # --- buttons -------------------------------------------------------
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._apply_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # slots
    # ------------------------------------------------------------------

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output Directory")
        if path:
            self._output_edit.setText(path)

    def _apply_and_accept(self) -> None:
        s = self._settings

        s.whisper_model = self._whisper_cb.currentText()
        s.output_dir = Path(self._output_edit.text())
        s.auto_upload = self._auto_upload_cb.isChecked()
        s.aws_region = self._aws_region.text()
        s.aws_bucket = self._aws_bucket.text()
        s.aws_access_key_id = self._aws_key.text()
        s.aws_secret_access_key = self._aws_secret.text()
        s.log_level = self._log_level_cb.currentText()
        s.log_file = Path(self._log_file.text())

        # audio devices
        mic_text = self._mic_cb.currentText()
        if mic_text == "(default)":
            s.microphone_device = None
        else:
            # extract index from format "name [index]"
            s.microphone_device = int(mic_text.rsplit("[", 1)[1].rstrip("]"))

        sys_text = self._sys_cb.currentText()
        if sys_text == "(default)":
            s.system_audio_device = None
        else:
            s.system_audio_device = int(sys_text.rsplit("[", 1)[1].rstrip("]"))

        self.accept()
