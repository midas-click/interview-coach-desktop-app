"""Audio capture via sounddevice — microphone and WASAPI loopback.

For system audio, uses ``sd.WasapiSettings(loopback=True)`` to capture
from WASAPI output devices. Falls back to the ``soundcard`` library if
the installed sounddevice doesn't support the loopback parameter.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AudioChunk:
    source: str          # "microphone" | "system_audio"
    data: np.ndarray     # float32, shape (n_samples,) — mono
    sample_rate: int     # Hz
    timestamp: float     # seconds since capture started


# ---------------------------------------------------------------------------
# device enumeration
# ---------------------------------------------------------------------------

def _host_name(hostapi_index: int) -> str:
    try:
        return sd.query_hostapis(hostapi_index)["name"]
    except Exception:
        return "Unknown"


def _wasapi_host_api_index() -> int | None:
    for idx, api in enumerate(sd.query_hostapis()):
        if "wasapi" in api["name"].lower():
            return idx
    return None


def list_microphones() -> list[dict[str, Any]]:
    """Real hardware microphone devices (deduped, best host API)."""
    all_devs = _query_all()
    mics = [d for d in all_devs if d["max_input_channels"] > 0 and _is_real_mic(d["name"])]
    return _deduplicate(mics)


def list_loopback_devices() -> list[dict[str, Any]]:
    """WASAPI output devices usable for system-audio loopback."""
    all_devs = _query_all()
    wasapi_idx = _wasapi_host_api_index()
    result: list[dict[str, Any]] = []
    for d in all_devs:
        if d["hostapi"] == wasapi_idx and d["max_output_channels"] > 0:
            result.append(d)
    return result


def test_device(device_index: int, as_loopback: bool = False, duration: float = 0.3) -> bool:
    """Test whether a device can be opened.

    When *as_loopback* is True, tries WASAPI loopback mode.
    """
    try:
        info = sd.query_devices(device_index)
        channels = max(info["max_input_channels"], 2) if not as_loopback else max(info["max_output_channels"], 2)
        rate = int(info["default_samplerate"])
        if rate <= 0:
            rate = 44100

        kwargs: dict = {
            "device": device_index,
            "channels": channels,
            "samplerate": rate,
            "dtype": "float32",
        }
        if as_loopback and _supports_wasapi_loopback():
            kwargs["extra_settings"] = sd.WasapiSettings(loopback=True)

        with contextlib.redirect_stderr(io.StringIO()):
            with sd.InputStream(**kwargs):
                sd.sleep(int(duration * 1000))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _supports_wasapi_loopback() -> bool:
    try:
        return "loopback" in inspect.signature(sd.WasapiSettings).parameters
    except Exception:
        return False


def _is_real_mic(name: str) -> bool:
    lower = name.lower()
    hints = ("microphone", "mic ", " mic", "headset", "line in", "line-in")
    return any(h in lower for h in hints)


def _query_all() -> list[dict[str, Any]]:
    try:
        sd_info = sd.query_devices()
    except Exception:
        return []
    devices: list[dict[str, Any]] = []
    for i, info in enumerate(sd_info):
        max_in = info["max_input_channels"]
        max_out = info["max_output_channels"]
        if max_in <= 0 and max_out <= 0:
            continue
        devices.append({
            "index": i,
            "name": info["name"],
            "hostapi": info["hostapi"],
            "host_api": _host_name(info["hostapi"]),
            "max_input_channels": max_in,
            "max_output_channels": max_out,
            "default_sample_rate": int(info["default_samplerate"]),
        })
    return devices


def _deduplicate(devices: list[dict]) -> list[dict]:
    import re
    host_rank = {"Windows WASAPI": 0, "Windows DirectSound": 1, "MME": 2}
    groups: dict[str, dict] = {}

    for d in devices:
        # Normalise MME truncation then strip all parentheticals
        name = d["name"].replace("(VB-Audio Virtual ", "(VB-Audio Virtual Cable)")
        name = re.sub(r"\s*\([^)]*$", "", name)
        name = re.sub(r"\s*\(.*?\)", "", name)
        name = re.sub(r"^\d+-\s*", "", name).strip().lower()
        key = name
        if key not in groups:
            groups[key] = d
        else:
            cur = host_rank.get(d["host_api"], 99)
            best = host_rank.get(groups[key]["host_api"], 99)
            if cur < best:
                groups[key] = d

    return sorted(groups.values(), key=lambda d: d["name"])


# ---------------------------------------------------------------------------
# shared audio-buffer helper
# ---------------------------------------------------------------------------

def _emit_chunk(
    indata: np.ndarray,
    sample_rate: int,
    chunk_duration: float,
    buffer: list[np.ndarray],
    lock: threading.Lock,
    audio_queue: queue.Queue[AudioChunk] | None,
    start_time: float,
    source: str,
) -> None:
    """Buffer audio until *chunk_duration* is reached, then emit an AudioChunk."""
    if audio_queue is None:
        return
    with lock:
        buffer.append(indata.copy())
        total = sum(c.shape[0] for c in buffer)
        needed = int(sample_rate * chunk_duration)
        if total < needed:
            return
        samples = np.concatenate(buffer, axis=0)
        buffer.clear()

    if samples.ndim == 2 and samples.shape[1] > 1:
        mono = samples.mean(axis=1).astype(np.float32)
    else:
        mono = samples.ravel().astype(np.float32)

    elapsed = time.monotonic() - start_time
    audio_queue.put(AudioChunk(source=source, data=mono, sample_rate=sample_rate, timestamp=elapsed))


# ---------------------------------------------------------------------------
# capture classes
# ---------------------------------------------------------------------------

class AudioCapture:
    """Captures from a single microphone device."""

    def __init__(
        self,
        source_label: str = "microphone",
        device_index: int | None = None,
        chunk_duration: float = 1.0,
    ) -> None:
        self._source = source_label
        self._device_index = device_index
        self._chunk_duration = chunk_duration
        self._stream: sd.InputStream | None = None
        self._queue: queue.Queue[AudioChunk] | None = None
        self._start_time: float = 0.0
        self._sample_rate: int = 0
        self._buffer: list[np.ndarray] = []
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._stream is not None and self._stream.active

    def start(self, target_queue: queue.Queue[AudioChunk]) -> None:
        if self.is_running:
            return
        self._queue = target_queue
        self._start_time = time.monotonic()
        self._buffer.clear()

        channels, rate = self._query_input_params()
        self._sample_rate = rate

        self._stream = sd.InputStream(
            device=self._device_index,
            channels=channels,
            samplerate=rate,
            dtype="float32",
            blocksize=int(rate * self._chunk_duration),
            callback=self._on_audio,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._queue = None
        with self._lock:
            self._buffer.clear()

    def _query_input_params(self) -> tuple[int, int]:
        channels = 1
        rate = 44100
        if self._device_index is not None:
            try:
                info = sd.query_devices(self._device_index)
                channels = max(info["max_input_channels"], 1)
                dr = int(info["default_samplerate"])
                if dr > 0:
                    rate = dr
            except Exception:
                pass
        return channels, rate

    def _on_audio(self, indata: np.ndarray, frames: int, _time: Any, status: int) -> None:
        del frames, status  # unused callback args
        _emit_chunk(indata, self._sample_rate, self._chunk_duration,
                    self._buffer, self._lock, self._queue,
                    self._start_time, self._source)


class LoopbackCapture:
    """Captures system audio via WASAPI loopback or soundcard fallback."""

    def __init__(
        self,
        device_index: int | None = None,
        chunk_duration: float = 1.0,
    ) -> None:
        self._device_index = device_index
        self._chunk_duration = chunk_duration
        self._stream: sd.InputStream | None = None
        self._queue: queue.Queue[AudioChunk] | None = None
        self._start_time: float = 0.0
        self._sample_rate: int = 0
        self._buffer: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._soundcard_recorder: Any = None
        self._soundcard_thread: threading.Thread | None = None
        self._soundcard_stop: threading.Event | None = None

    @property
    def is_running(self) -> bool:
        if self._stream is not None and self._stream.active:
            return True
        return self._soundcard_thread is not None and self._soundcard_thread.is_alive()

    def start(self, target_queue: queue.Queue[AudioChunk]) -> None:
        if self.is_running:
            return

        self._queue = target_queue
        self._start_time = time.monotonic()
        self._buffer.clear()

        if _supports_wasapi_loopback():
            self._start_wasapi_loopback()
        else:
            self._start_soundcard_loopback()

    def _start_wasapi_loopback(self) -> None:
        device = self._resolve_loopback_device()
        channels = max(device.get("max_output_channels", 2), 2)
        rate = int(device.get("default_sample_rate", 0) or 44100)
        self._sample_rate = rate

        self._stream = sd.InputStream(
            device=device["index"],
            channels=channels,
            samplerate=rate,
            dtype="float32",
            blocksize=int(rate * self._chunk_duration),
            extra_settings=sd.WasapiSettings(loopback=True),
            callback=self._on_audio,
        )
        self._stream.start()

    def _start_soundcard_loopback(self) -> None:
        import soundcard as sc
        speaker = sc.default_speaker()
        mics = sc.all_microphones(include_loopback=True)
        loopback = None
        for mic in mics:
            if mic.name == speaker.name:
                loopback = mic
                break
        if loopback is None:
            for mic in mics:
                if speaker.name.lower() in mic.name.lower() or mic.name.lower() in speaker.name.lower():
                    loopback = mic
                    break
        if loopback is None:
            raise RuntimeError(
                f"No loopback microphone found for speaker '{speaker.name}'. "
                f"Install a newer sounddevice for WASAPI loopback support."
            )
        rate = 44100
        self._sample_rate = rate
        # Create recorder on the main thread (COM must be initialised)
        self._soundcard_recorder = loopback.recorder(samplerate=rate).__enter__()
        self._soundcard_stop = threading.Event()
        self._soundcard_thread = threading.Thread(
            target=self._run_soundcard, args=(rate,),
            daemon=True, name="loopback-soundcard",
        )
        self._soundcard_thread.start()

    def _run_soundcard(self, rate: int) -> None:
        recorder = self._soundcard_recorder
        try:
            while recorder and not self._soundcard_stop.is_set():  # type: ignore[union-attr]
                chunk_frames = int(rate * self._chunk_duration)
                samples = recorder.record(numframes=chunk_frames)
                if not samples.size:
                    continue
                if samples.ndim == 2 and samples.shape[1] > 1:
                    mono = samples.mean(axis=1).astype(np.float32)
                else:
                    mono = samples.ravel().astype(np.float32)
                elapsed = time.monotonic() - self._start_time
                try:
                    if self._queue is not None:
                        self._queue.put(AudioChunk(
                            source="system_audio", data=mono,
                            sample_rate=rate, timestamp=elapsed,
                        ))
                except queue.Full:
                    pass
        except Exception:
            pass  # normal during shutdown

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # Signal soundcard thread to stop, then close recorder on main thread
        if self._soundcard_stop is not None:
            self._soundcard_stop.set()
        if self._soundcard_thread and self._soundcard_thread.is_alive():
            self._soundcard_thread.join(timeout=3)
            self._soundcard_thread = None
        if self._soundcard_recorder is not None:
            try:
                self._soundcard_recorder.__exit__(None, None, None)
            except Exception:
                pass
            self._soundcard_recorder = None
        self._soundcard_stop = None
        self._queue = None
        with self._lock:
            self._buffer.clear()

    def _resolve_loopback_device(self) -> dict:
        wasapi_idx = _wasapi_host_api_index()
        if wasapi_idx is None:
            raise RuntimeError("WASAPI host API not found")

        devices = sd.query_devices()
        requested = self._device_index

        # If a specific device was requested
        if requested is not None:
            info = devices[requested]
            if info["hostapi"] != wasapi_idx:
                raise RuntimeError(f"Device {requested} is not a WASAPI output device")
            if info["max_output_channels"] <= 0:
                raise RuntimeError(f"Device {requested} has no output channels")
            return {"index": requested, **dict(info)}

        # Try default output device if it's WASAPI
        default_out = sd.default.device[1]
        if default_out is not None and default_out >= 0:
            info = devices[default_out]
            if info["hostapi"] == wasapi_idx and info["max_output_channels"] > 0:
                return {"index": default_out, **dict(info)}

        # Fallback: first WASAPI output device
        for idx, info in enumerate(devices):
            if info["hostapi"] == wasapi_idx and info["max_output_channels"] > 0:
                return {"index": idx, **dict(info)}

        raise RuntimeError("No WASAPI output device found for loopback")

    def _on_audio(self, indata: np.ndarray, frames: int, _time: Any, status: int) -> None:
        del frames, status
        _emit_chunk(indata, self._sample_rate, self._chunk_duration,
                    self._buffer, self._lock, self._queue,
                    self._start_time, "system_audio")
