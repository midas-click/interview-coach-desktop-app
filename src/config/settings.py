"""Application settings: typed model + YAML persistence."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

VALID_WHISPER_MODELS = frozenset({
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3",
    "distil-small.en", "distil-medium.en", "distil-large-v2",
    "distil-large-v3",
})

VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class Settings(BaseModel):
    """User-configurable application settings.

    AWS secrets are stored in this model for persistence across sessions,
    but environment variables (AWS_ACCESS_KEY_ID, etc.) take precedence
    at load time.
    """

    # ---------- transcription ----------
    whisper_model: str = "base"

    # ---------- audio ----------
    microphone_device: int | None = Field(
        default=None,
        description="PyAudio device index for microphone input",
    )
    system_audio_device: int | None = Field(
        default=None,
        description="PyAudio device index for system audio (loopback)",
    )

    # ---------- output ----------
    output_dir: Path = Field(
        default=Path("./output"),
        description="Directory for local transcript exports",
    )
    auto_upload: bool = Field(
        default=True,
        description="Upload to S3 automatically when interview finishes",
    )

    # ---------- aws ----------
    aws_region: str = "us-east-1"
    aws_bucket: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # ---------- logging ----------
    log_level: str = "INFO"
    log_file: Path = Field(default=Path("./logs/transcriber.log"))

    # ---------- validation ----------

    @field_validator("whisper_model")
    @classmethod
    def _check_whisper_model(cls, v: str) -> str:
        # Accept HuggingFace model names OR local paths (e.g. ./models/whisper-base)
        if Path(v).exists():
            return str(Path(v).resolve())
        if v not in VALID_WHISPER_MODELS:
            raise ValueError(
                f"Unknown whisper model '{v}'. "
                f"Valid models: {', '.join(sorted(VALID_WHISPER_MODELS))}"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, v: str) -> str:
        upper = v.upper()
        if upper not in VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log level '{v}'. "
                f"Valid levels: {', '.join(sorted(VALID_LOG_LEVELS))}"
            )
        return upper


# ---------------------------------------------------------------------------
# persistence helpers
# ---------------------------------------------------------------------------

def load_settings(path: Path) -> Settings:
    """Load settings from a YAML file.

    Falls back to defaults if the file is missing.
    """
    if not path.exists():
        return Settings()

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    settings = Settings(**raw)
    _apply_env_overrides(settings)
    return settings


def save_settings(settings: Settings, path: Path) -> None:
    """Persist settings to a YAML file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = settings.model_dump(mode="json")
    path.write_text(yaml.safe_dump(data, default_flow_style=False), encoding="utf-8")


def _apply_env_overrides(settings: Settings) -> None:
    """Override AWS fields from environment variables when set."""
    for key in ("aws_access_key_id", "aws_secret_access_key", "aws_region", "aws_bucket"):
        env_key = key.upper()
        if env_key in os.environ:
            setattr(settings, key, os.environ[env_key])
