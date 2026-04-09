"""Cross-platform audio loopback capture via sounddevice.

- Windows: WASAPI loopback (built-in, zero setup)
- Linux: PulseAudio/PipeWire monitor source (built-in)
- macOS: BlackHole virtual audio device (requires `brew install blackhole-2ch`)
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

logger = logging.getLogger("treadpal.audio")

# sounddevice returns device/hostapi info as dicts with Any values
type DeviceInfo = dict[str, Any]


def find_loopback_device() -> int:
    """Find a loopback/monitor audio input device for the current platform."""
    devices: list[DeviceInfo] = sd.query_devices()  # type: ignore[assignment]
    hostapis: list[DeviceInfo] = sd.query_hostapis()  # type: ignore[assignment]

    if sys.platform == "win32":
        return _find_wasapi_loopback(devices, hostapis)
    elif sys.platform == "linux":
        return _find_pulse_monitor(devices)
    elif sys.platform == "darwin":
        return _find_blackhole(devices)
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")


def _find_wasapi_loopback(
    devices: list[DeviceInfo], hostapis: list[DeviceInfo]
) -> int:
    """Find WASAPI loopback device on Windows."""
    wasapi_index: int | None = None
    for i, api in enumerate(hostapis):
        if "WASAPI" in str(api.get("name", "")):
            wasapi_index = i
            break
    if wasapi_index is None:
        raise RuntimeError("WASAPI host API not found")

    for i, dev in enumerate(devices):
        if (
            dev.get("hostapi") == wasapi_index
            and int(dev.get("max_input_channels", 0)) > 0
            and "loopback" in str(dev.get("name", "")).lower()
        ):
            logger.info("Using WASAPI loopback device: %s", dev.get("name"))
            return i

    default_output = hostapis[wasapi_index].get("default_output_device")
    if default_output is not None:
        idx = int(default_output)
        logger.info(
            "Using WASAPI default output as loopback: %s",
            devices[idx].get("name"),
        )
        return idx

    raise RuntimeError("No WASAPI loopback device found")


def _find_pulse_monitor(devices: list[DeviceInfo]) -> int:
    """Find PulseAudio/PipeWire monitor source on Linux."""
    for i, dev in enumerate(devices):
        name = str(dev.get("name", "")).lower()
        if "monitor" in name and int(dev.get("max_input_channels", 0)) > 0:
            logger.info("Using PulseAudio monitor: %s", dev.get("name"))
            return i

    raise RuntimeError(
        "No PulseAudio/PipeWire monitor device found. "
        "Ensure PulseAudio or PipeWire is running."
    )


def _find_blackhole(devices: list[DeviceInfo]) -> int:
    """Find BlackHole virtual audio device on macOS."""
    for i, dev in enumerate(devices):
        name = str(dev.get("name", "")).lower()
        if "blackhole" in name and int(dev.get("max_input_channels", 0)) > 0:
            logger.info("Using BlackHole device: %s", dev.get("name"))
            return i

    raise RuntimeError(
        "BlackHole audio device not found. "
        "Install it with: brew install blackhole-2ch\n"
        "Then create a Multi-Output Device in Audio MIDI Setup."
    )


class AudioCapture:
    """Captures system audio via platform loopback and feeds an asyncio queue."""

    def __init__(
        self,
        sample_rate: int = 44100,
        hop_size: int = 256,
        device_index: int | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._hop_size = hop_size
        self._device_index = device_index
        self._buffer: asyncio.Queue[NDArray[np.float32]] = asyncio.Queue(maxsize=64)
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        """Start the audio capture stream."""
        self._loop = asyncio.get_running_loop()
        device = self._device_index if self._device_index is not None else find_loopback_device()

        extra_settings = None
        if sys.platform == "win32":
            extra_settings = sd.WasapiSettings(auto_convert=True)

        self._stream = sd.InputStream(
            device=device,
            channels=1,
            samplerate=self._sample_rate,
            blocksize=self._hop_size,
            dtype=np.float32,
            callback=self._audio_callback,
            extra_settings=extra_settings,
        )
        self._stream.start()
        logger.info("Audio capture started (device=%s)", device)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Audio capture stopped")

    def _audio_callback(
        self,
        indata: NDArray[np.float32],
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Called from PortAudio thread — must not block."""
        if self._loop is not None:
            mono = indata[:, 0].copy()
            try:
                self._loop.call_soon_threadsafe(self._buffer.put_nowait, mono)
            except asyncio.QueueFull:
                pass  # Drop frame if consumer is too slow

    async def read_block(self) -> NDArray[np.float32]:
        """Async read of next audio block."""
        return await self._buffer.get()
