# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.0",
#     "websockets>=13",
#     "PyAudioWPatch>=0.2.12; sys_platform == 'win32'",
#     "sounddevice>=0.5; sys_platform != 'win32'",
# ]
# ///
"""Thin audio streamer — captures system audio and streams to TreadPal server via WebSocket.

All BPM detection happens server-side. This script just captures and forwards audio.

    uv run bpm_agent.py --server ws://192.168.1.50:8080/ws/audio
    uv run bpm_agent.py --list-devices
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

import numpy as np
import websockets
from numpy.typing import NDArray

logger = logging.getLogger("bpm_agent")


# --- Audio capture ---

def _find_and_start_capture(device_index: int | None, queue: asyncio.Queue[bytes], loop: asyncio.AbstractEventLoop) -> tuple[object, int]:
    """Start audio capture, return (handle, sample_rate)."""
    if sys.platform == "win32":
        return _start_wasapi(device_index, queue, loop)
    else:
        return _start_sounddevice(device_index, queue, loop)


def _start_wasapi(device_index: int | None, queue: asyncio.Queue[bytes], loop: asyncio.AbstractEventLoop) -> tuple[tuple[object, object], int]:
    import pyaudiowpatch as pyaudio  # type: ignore[import-untyped]

    p = pyaudio.PyAudio()
    if device_index is not None:
        info = p.get_device_info_by_index(device_index)
    else:
        info = p.get_default_wasapi_loopback()
        if not info:
            raise RuntimeError("No WASAPI loopback device found")

    sr = int(info.get("defaultSampleRate", 44100))
    channels = int(info.get("maxInputChannels", 2))
    logger.info("WASAPI loopback: %s (%dHz, %dch)", info.get("name"), sr, channels)

    def callback(in_data: bytes | None, frame_count: int, time_info: object, status: int) -> tuple[None, int]:
        if in_data:
            samples = np.frombuffer(in_data, dtype=np.float32)
            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1).astype(np.float32)
            try:
                loop.call_soon_threadsafe(queue.put_nowait, samples.tobytes())
            except asyncio.QueueFull:
                pass
        return (None, pyaudio.paContinue)

    stream = p.open(
        format=pyaudio.paFloat32, channels=channels, rate=sr,
        input=True, frames_per_buffer=2048,
        input_device_index=int(info["index"]),
        stream_callback=callback,
    )
    return (stream, p), sr


def _start_sounddevice(device_index: int | None, queue: asyncio.Queue[bytes], loop: asyncio.AbstractEventLoop) -> tuple[object, int]:
    import sounddevice as sd

    device = device_index
    sr = 44100
    if device is None:
        devs: list[dict] = sd.query_devices()  # type: ignore[assignment]
        if sys.platform == "darwin":
            for i, d in enumerate(devs):
                if "blackhole" in str(d.get("name", "")).lower() and int(d.get("max_input_channels", 0)) > 0:
                    device = i
                    break
            if device is None:
                raise RuntimeError("BlackHole not found. Install: brew install blackhole-2ch")
        else:
            for i, d in enumerate(devs):
                if "monitor" in str(d.get("name", "")).lower() and int(d.get("max_input_channels", 0)) > 0:
                    device = i
                    break
            if device is None:
                raise RuntimeError("No PulseAudio/PipeWire monitor found.")

    def callback(indata: NDArray[np.float32], frames: int, time_info: object, status: object) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, indata[:, 0].copy().tobytes())
        except asyncio.QueueFull:
            pass

    stream = sd.InputStream(
        device=device, channels=1, samplerate=sr,
        blocksize=2048, dtype=np.float32, callback=callback,
    )
    stream.start()
    return stream, sr


def stop_capture(handle: object) -> None:
    if sys.platform == "win32":
        stream, pa = handle  # type: ignore[misc]
        stream.stop_stream()
        stream.close()
        pa.terminate()
    else:
        handle.stop()  # type: ignore[union-attr]
        handle.close()  # type: ignore[union-attr]


# --- Device listing ---

def list_devices() -> None:
    if sys.platform == "win32":
        import pyaudiowpatch as pyaudio  # type: ignore[import-untyped]
        p = pyaudio.PyAudio()
        print("Audio devices (input):")
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d.get("maxInputChannels", 0) > 0:
                tag = " [LOOPBACK]" if d.get("isLoopbackDevice", False) else ""
                print(f"  {i:3d}: {d['name']}  ({int(d['defaultSampleRate'])}Hz){tag}")
        try:
            default = p.get_default_wasapi_loopback()
            if default:
                print(f"\n  Auto-detected: {default['index']} — {default['name']}")
        except Exception:
            pass
        p.terminate()
    else:
        import sounddevice as sd
        devs = sd.query_devices()
        apis = sd.query_hostapis()
        print("Audio devices (input):")
        for i, d in enumerate(devs):
            if d["max_input_channels"] > 0:  # type: ignore[operator]
                api = apis[d["hostapi"]]["name"]  # type: ignore[index]
                print(f"  {i:3d}: {d['name']}  ({api})")


# --- Main ---

async def run(server_url: str, device_index: int | None) -> None:
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=128)
    loop = asyncio.get_event_loop()
    handle, sr = _find_and_start_capture(device_index, queue, loop)

    ws_url = f"{server_url}?sr={sr}"
    logger.info("Connecting to %s", ws_url)

    try:
        async for ws in websockets.connect(ws_url):
            logger.info("Connected, streaming audio (%dHz)", sr)
            try:
                # Send audio and receive status concurrently
                chunks_sent = 0

                async def send_audio() -> None:
                    nonlocal chunks_sent
                    while True:
                        data = await queue.get()
                        await ws.send(data)
                        chunks_sent += 1
                        if chunks_sent % 100 == 0:
                            kb = chunks_sent * len(data) / 1024
                            logger.info("Streamed %d chunks (%.0f KB)", chunks_sent, kb)

                async def recv_status() -> None:
                    async for msg in ws:
                        if isinstance(msg, str):
                            status = json.loads(msg)
                            parts = [f"BPM={status.get('bpm', '?')}"]
                            parts.append(f"speed={status.get('ramped_speed_kmh', '?')} km/h")
                            parts.append(f"x{status.get('harmonic', '?')}")
                            if "incline_pct" in status:
                                parts.append(f"incline={status['incline_pct']}%")
                            logger.info(" | ".join(parts))

                await asyncio.gather(send_audio(), recv_status())
            except websockets.ConnectionClosed:
                logger.warning("Connection lost, reconnecting...")
                continue
    except KeyboardInterrupt:
        pass
    finally:
        stop_capture(handle)
        logger.info("Stopped")


def main() -> None:
    p = argparse.ArgumentParser(description="TreadPal Audio Streamer")
    p.add_argument("--server", default="ws://127.0.0.1:8080/ws/audio",
                    help="WebSocket URL (default: ws://127.0.0.1:8080/ws/audio)")
    p.add_argument("--device", type=int, default=None, help="Audio device index (--list-devices)")
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.list_devices:
        list_devices()
        return

    asyncio.run(run(args.server, args.device))


if __name__ == "__main__":
    main()
