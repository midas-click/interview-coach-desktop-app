"""Application settings loaded from environment / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

VALID_WHISPER_MODELS = frozenset({
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3", "large-v3-turbo", "turbo",
    "distil-small.en", "distil-medium.en", "distil-large-v2",
    "distil-large-v3",
})

VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class Settings(BaseSettings):
    """Application settings loaded from .env / environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---------- transcription ----------
    whisper_model: str = "base"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"

    # ---------- vad ----------
    vad_threshold: float = 0.5
    vad_silence_ms: int = 800
    vad_min_speech_ms: int = 500

    # ---------- output ----------
    output_dir: Path = Field(
        default=Path("./output"),
        description="Directory for local transcript exports",
    )
    temp_dir: Path = Field(
        default=Path("./data/temp"),
        description="Directory for temporary WAV files during transcription",
    )

    # ---------- aws ----------
    aws_region: str = "us-east-2"
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
