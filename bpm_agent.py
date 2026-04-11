# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.0",
#     "websockets>=13",
#     "httpx>=0.27",
#     "rich>=13",
#     "PyAudioWPatch>=0.2.12; sys_platform == 'win32'",
#     "sounddevice>=0.5; sys_platform != 'win32'",
# ]
# ///
"""TreadPal BPM Agent — streams audio to server, shows live TUI.

    uv run bpm_agent.py --server ws://192.168.1.50:8080/ws/audio
    uv run bpm_agent.py --list-devices

Keys: [u] harmonic up  [d] harmonic down  [r] reset to auto  [q] quit
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from collections import deque

import httpx
import numpy as np
import websockets
from numpy.typing import NDArray
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text


# --- Non-blocking key reader ---

class KeyReader:
    """Cross-platform non-blocking single-char reader."""

    def __init__(self) -> None:
        self._queue: deque[str] = deque(maxlen=16)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if sys.platform == "win32":
            import msvcrt
            while not self._stop.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    self._queue.append(ch.lower())
                time.sleep(0.05)
        else:
            import tty
            import termios
            import select
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while not self._stop.is_set():
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        ch = sys.stdin.read(1)
                        self._queue.append(ch.lower())
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def get(self) -> str | None:
        return self._queue.popleft() if self._queue else None

    def stop(self) -> None:
        self._stop.set()


# --- Audio capture ---

def start_capture(device_index: int | None, queue: asyncio.Queue[bytes], loop: asyncio.AbstractEventLoop) -> tuple[object, int]:
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


# --- TUI ---

class TUI:
    def __init__(self, server_http: str) -> None:
        self.server_http = server_http
        self.bpm: float = 0
        self.target_speed: float = 0
        self.ramped_speed: float = 0
        self.harmonic: float = 0
        self.harmonic_override: bool = False
        self.stride: float = 0
        self.incline: float | None = None
        self.paused: bool = False
        self.connected: bool = False
        self.bpm_history: deque[float] = deque(maxlen=30)
        self.speed_history: deque[float] = deque(maxlen=30)
        self._http = httpx.Client(timeout=2.0)

    def update(self, status: dict) -> None:
        self.bpm = status.get("bpm", 0)
        self.target_speed = status.get("target_speed_kmh", 0)
        self.ramped_speed = status.get("ramped_speed_kmh", 0)
        self.harmonic = status.get("harmonic", 0)
        self.harmonic_override = status.get("harmonic_override", False)
        self.stride = status.get("stride_m", 0)
        self.incline = status.get("incline_pct")
        self.paused = status.get("paused", False)
        self.bpm_history.append(self.bpm)
        self.speed_history.append(self.ramped_speed)

    def handle_key(self, key: str) -> bool:
        """Handle keypress. Returns False if should quit."""
        try:
            if key == "q":
                return False
            elif key == "u":
                self._http.post(f"{self.server_http}/api/bpm/harmonic/up")
            elif key == "d":
                self._http.post(f"{self.server_http}/api/bpm/harmonic/down")
            elif key == "r":
                self._http.post(f"{self.server_http}/api/bpm/harmonic/reset")
            elif key == "p":
                endpoint = "/api/bpm/resume" if self.paused else "/api/bpm/pause"
                self._http.post(f"{self.server_http}{endpoint}")
        except httpx.RequestError:
            pass
        return True

    def render(self) -> Table:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(justify="right", style="bold")
        grid.add_column()

        grid.add_row("BPM", f"[cyan bold]{self.bpm:.1f}[/]")

        h_style = "[yellow bold]" if self.harmonic_override else "[green]"
        h_label = f"{h_style}{self.harmonic:.2f}x[/]"
        if self.harmonic_override:
            h_label += " [yellow](override)[/]"
        grid.add_row("Harmonic", h_label)

        grid.add_row("Target", f"{self.target_speed:.2f} km/h")
        grid.add_row("Speed", f"[bold]{self.ramped_speed:.2f}[/] km/h")
        grid.add_row("Stride", f"{self.stride:.3f} m")

        if self.incline is not None:
            grid.add_row("Incline", f"{self.incline:.1f}%")

        # Sparkline for BPM trend
        if len(self.bpm_history) > 1:
            bpm_spark = _sparkline(list(self.bpm_history))
            grid.add_row("BPM trend", Text(bpm_spark, style="dim cyan"))

        # Sparkline for speed trend
        if len(self.speed_history) > 1:
            speed_spark = _sparkline(list(self.speed_history))
            grid.add_row("Speed trend", Text(speed_spark, style="dim green"))

        if self.paused:
            status_str = "[yellow]Paused[/]"
        elif self.connected:
            status_str = "[green]Connected[/]"
        else:
            status_str = "[red]Disconnected[/]"
        grid.add_row("Status", status_str)
        grid.add_row("", Text("(u)p  (d)own  (r)eset  (p)ause  (q)uit", style="dim"))

        return grid


def _sparkline(values: list[float]) -> str:
    if not values:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    rng = hi - lo if hi > lo else 1.0
    return "".join(blocks[min(8, int((v - lo) / rng * 8))] for v in values)


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
                print(f"\n  Auto-detected: {default['index']} -- {default['name']}")
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

async def run(ws_url: str, http_url: str, device_index: int | None) -> None:
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=128)
    loop = asyncio.get_event_loop()
    handle, sr = start_capture(device_index, audio_queue, loop)
    keys = KeyReader()
    tui = TUI(http_url)
    console = Console()

    full_url = f"{ws_url}?sr={sr}"

    try:
        with Live(tui.render(), console=console, refresh_per_second=4, screen=True) as live:
            async for ws in websockets.connect(full_url):
                tui.connected = True
                live.update(tui.render())

                try:
                    async def send_audio() -> None:
                        while True:
                            data = await audio_queue.get()
                            await ws.send(data)

                    async def recv_status() -> None:
                        async for msg in ws:
                            if isinstance(msg, str):
                                tui.update(json.loads(msg))
                                live.update(tui.render())

                    async def check_keys() -> None:
                        while True:
                            key = keys.get()
                            if key is not None:
                                if not tui.handle_key(key):
                                    raise KeyboardInterrupt
                            await asyncio.sleep(0.05)

                    await asyncio.gather(send_audio(), recv_status(), check_keys())
                except websockets.ConnectionClosed:
                    tui.connected = False
                    live.update(tui.render())
                    continue
    except KeyboardInterrupt:
        pass
    finally:
        keys.stop()
        stop_capture(handle)


def main() -> None:
    p = argparse.ArgumentParser(description="TreadPal BPM Agent")
    p.add_argument("--server", default="ws://127.0.0.1:8080/ws/audio",
                    help="WebSocket URL (default: ws://127.0.0.1:8080/ws/audio)")
    p.add_argument("--device", type=int, default=None, help="Audio device index (--list-devices)")
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        list_devices()
        return

    # Derive HTTP URL from WS URL for API calls
    http_url = args.server.replace("ws://", "http://").replace("wss://", "https://")
    http_url = http_url.split("/ws/")[0]

    asyncio.run(run(args.server, http_url, args.device))


if __name__ == "__main__":
    main()
