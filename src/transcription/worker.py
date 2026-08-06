"""Background transcription worker — polls SQLite for queued utterances, 
transcribes them one at a time, and writes results back.

Runs in its own thread. Completely independent from audio capture.
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from src.logger.logger import get_logger

if TYPE_CHECKING:
    from src.storage.repository import Repository
    from src.transcription.engine import TranscriptionEngine

log = get_logger(__name__)

POLL_INTERVAL = 0.5


class TranscriptionWorker:
    """Polls the utterances table for ``queued`` rows and transcribes each one."""

    def __init__(self, repo: Repository, engine: TranscriptionEngine) -> None:
        self._repo = repo
        self._engine = engine
        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="transcriber")
        self._thread.start()
        log.info("Transcription worker started")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=60.0)
        log.info("Transcription worker stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._loop())
        except Exception:
            log.exception("Worker event loop crashed")
        finally:
            loop.close()

    async def _loop(self) -> None:
        log.info("Worker loop started — polling for queued utterances")
        idle_polls = 0

        while self._running:
            try:
                utterance = await self._repo.next_queued()
            except Exception:
                log.exception("Failed to query queued utterances")
                await asyncio.sleep(POLL_INTERVAL)
                continue

            if utterance is None:
                idle_polls += 1
                if idle_polls % 20 == 1:  # log every ~10s
                    log.debug("Worker idle — no queued utterances (%d polls)", idle_polls)
                await asyncio.sleep(POLL_INTERVAL)
                continue

            idle_polls = 0
            log.info("Worker picked up utterance %d from queue (%s, %d–%d ms, wav=%s)",
                     utterance.id, utterance.speaker, utterance.start_ms,
                     utterance.end_ms, Path(utterance.audio_path).name)

            # Verify WAV exists before attempting transcription
            if not os.path.isfile(utterance.audio_path):
                log.error("WAV file missing for utterance %d: %s", utterance.id, utterance.audio_path)
                try:
                    await self._repo.mark_failed(utterance.id)
                except Exception:
                    log.exception("Failed to mark utterance %d as failed", utterance.id)
                continue

            try:
                await self._repo.mark_processing(utterance.id)
            except Exception:
                log.exception("Failed to mark utterance %d as processing", utterance.id)
                continue

            try:
                segments = await asyncio.to_thread(
                    self._engine.transcribe, utterance.audio_path,
                )
            except Exception:
                log.exception("Transcription crashed for utterance %d", utterance.id)
                try:
                    await self._repo.mark_failed(utterance.id)
                except Exception:
                    log.exception("Failed to mark utterance %d as failed", utterance.id)
                continue

            if not segments:
                log.info("Utterance %d — no speech detected, discarding", utterance.id)
                try:
                    await self._repo.mark_failed(utterance.id)
                except Exception:
                    log.exception("Failed to mark utterance %d as failed", utterance.id)
                try:
                    Path(utterance.audio_path).unlink(missing_ok=True)
                except Exception:
                    pass
                continue

            text = " ".join(s.text for s in segments)
            confidence = round(
                sum(s.confidence for s in segments) / len(segments), 3
            )

            try:
                await self._repo.mark_completed(utterance.id, transcript=text, confidence=confidence)
            except Exception:
                log.exception("Failed to mark utterance %d as completed", utterance.id)
                continue

            # delete temp WAV
            try:
                Path(utterance.audio_path).unlink(missing_ok=True)
            except Exception:
                pass

            log.info("Utterance %d completed (confidence=%.3f, text=%s)",
                     utterance.id, confidence, text[:80])
