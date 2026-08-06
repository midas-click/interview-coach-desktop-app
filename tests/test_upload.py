"""Tests for S3 uploader."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from src.config.settings import Settings
from src.upload.s3_uploader import S3Uploader, UploadError


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        aws_region="us-east-2",
        aws_bucket="test-bucket",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        output_dir=tmp_path / "output",
    )


@pytest.fixture
def mock_boto() -> MagicMock:
    with patch("src.upload.s3_uploader.boto3.Session") as mock:
        yield mock


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------

def test_upload_saves_local_and_calls_s3(settings: Settings, mock_boto: MagicMock) -> None:
    uploader = S3Uploader(settings)
    data = {
        "schemaVersion": 1,
        "interviewId": "abc",
        "utterances": [],
    }

    import asyncio

    async def _test():
        path = await uploader.upload(data, "abc")
        assert path.exists()
        assert json.loads(path.read_text()) == data

        mock_client = mock_boto.return_value.client.return_value
        assert mock_client.upload_file.call_count == 1
        args = mock_client.upload_file.call_args[0]
        assert args[1] == "test-bucket"
        assert args[2] == "interviews/abc/transcript.json"

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# retry logic
# ---------------------------------------------------------------------------

def test_upload_retries_on_failure(settings: Settings, mock_boto: MagicMock) -> None:
    uploader = S3Uploader(settings)
    mock_client = mock_boto.return_value.client.return_value

    error_response = {"Error": {"Code": "500", "Message": "Server Error"}}
    mock_client.upload_file.side_effect = [
        ClientError(error_response, "UploadPart"),
        ClientError(error_response, "UploadPart"),
        None,
    ]

    import asyncio

    async def _test():
        await uploader.upload({"interviewId": "x", "utterances": []}, "x")
        assert mock_client.upload_file.call_count == 3

    asyncio.run(_test())


def test_upload_raises_after_max_retries(settings: Settings, mock_boto: MagicMock) -> None:
    uploader = S3Uploader(settings)
    mock_client = mock_boto.return_value.client.return_value
    mock_client.upload_file.side_effect = ClientError(
        {"Error": {"Code": "500", "Message": "Boom"}}, "UploadPart"
    )

    import asyncio

    async def _test():
        with pytest.raises(UploadError, match="Upload failed for interview x"):
            await uploader.upload({"interviewId": "x", "utterances": []}, "x")
        assert mock_client.upload_file.call_count == 3

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# local file survives upload failure
# ---------------------------------------------------------------------------

def test_local_file_kept_on_upload_failure(settings: Settings, mock_boto: MagicMock) -> None:
    uploader = S3Uploader(settings)
    mock_boto.return_value.client.return_value.upload_file.side_effect = ClientError(
        {"Error": {"Code": "500", "Message": "Boom"}}, "UploadPart"
    )

    import asyncio

    async def _test():
        with pytest.raises(UploadError):
            await uploader.upload({"interviewId": "keep-me", "utterances": []}, "keep-me")

        local = settings.output_dir / "keep-me" / "transcript.json"
        assert local.exists()

    asyncio.run(_test())
