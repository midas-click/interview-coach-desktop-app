"""AWS S3 uploader with automatic retry and local-fallback safety.

Uploads are performed in a thread executor so the UI never freezes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
from botocore.exceptions import BotoCoreError, ClientError

if TYPE_CHECKING:
    from src.config.settings import Settings


class UploadError(Exception):
    """Raised when an upload fails after all retries."""


MAX_RETRIES = 3
BASE_DELAY = 1.0


class S3Uploader:
    """Uploads transcript JSON to S3 with exponential backoff."""

    def __init__(self, settings: Settings) -> None:
        self._bucket = settings.aws_bucket
        self._region = settings.aws_region
        self._output_dir = settings.output_dir
        ak = settings.aws_access_key_id
        sk = settings.aws_secret_access_key

        session_kwargs: dict = {"region_name": self._region}
        if ak and sk:
            session_kwargs["aws_access_key_id"] = ak
            session_kwargs["aws_secret_access_key"] = sk

        self._session = boto3.Session(**session_kwargs)
        self._client = self._session.client("s3")

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def upload(self, export_data: dict, interview_id: str) -> Path:
        """Export JSON to local disk, then upload to S3.

        Returns the path to the local JSON file (kept even if upload fails).
        Raises ``UploadError`` only when S3 fails after all retries.
        """
        local_path = self._save_local(export_data, interview_id)

        try:
            await self._upload_with_retry(local_path, interview_id)
        except Exception:
            raise UploadError(
                f"Upload failed for interview {interview_id}. "
                f"Local copy kept at {local_path}"
            )

        return local_path

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _save_local(self, data: dict, interview_id: str) -> Path:
        dir_path = self._output_dir / interview_id
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / "transcript.json"
        file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return file_path

    async def _upload_with_retry(self, file_path: Path, interview_id: str) -> None:
        s3_key = f"interviews/{interview_id}/transcript.json"
        last_exc: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                await asyncio.to_thread(
                    self._client.upload_file,
                    str(file_path),
                    self._bucket,
                    s3_key,
                )
                return
            except (BotoCoreError, ClientError) as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(BASE_DELAY * (2 ** (attempt - 1)))

        raise UploadError(
            f"S3 upload failed after {MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc
