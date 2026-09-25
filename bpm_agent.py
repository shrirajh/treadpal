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

Alongside the raw stereo audio (for server-side beat detection and stem
separation into vocals/other/bass/drums) it sends ~40 fps
visualiser frames: log spectrum split into bass/mid/high, zone energies,
a 12-note chroma with the dominant note, and kick/bass onsets.

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
            # Stereo (interleaved) for the server's stem separator; mono is duplicated
            samples = np.frombuffer(in_data, dtype=np.float32).reshape(-1, channels)
            samples = samples[:, :2] if channels >= 2 else np.repeat(samples, 2, axis=1)
            samples = np.ascontiguousarray(samples)
            try:
                loop.call_soon_threadsafe(queue.put_nowait, samples.tobytes())
            except asyncio.QueueFull:
                pass
        return (None, pyaudio.paContinue)

    stream = p.open(
        format=pyaudio.paFloat32, channels=channels, rate=sr,
        input=True, frames_per_buffer=1024,
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
            stereo = indata[:, :2] if indata.shape[1] >= 2 else np.repeat(indata[:, :1], 2, axis=1)
            loop.call_soon_threadsafe(queue.put_nowait, np.ascontiguousarray(stereo).tobytes())
        except asyncio.QueueFull:
            pass

    stream = sd.InputStream(
        device=device, channels=min(2, int(sd.query_devices(device)["max_input_channels"])), samplerate=sr,
        blocksize=1024, dtype=np.float32, callback=callback,
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


# --- Visualiser analysis ---

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class Analyzer:
    """Spectrum, bass/mid/high energy, chroma and onsets from a rolling window.

    Long window (8192) for frequency detail (bass notes are ~4 Hz apart);
    short window (2048) for onset timing, which a long window would smear.
    """

    # (low Hz, high Hz, bands) per zone: bass, mid, high
    ZONES = ((30.0, 250.0, 12), (250.0, 4000.0, 20), (4000.0, 16000.0, 12))

    def __init__(self, sr: int, n_fft: int = 8192, n_short: int = 2048) -> None:
        self.sr = sr
        self.n = n_fft
        self.n_short = n_short
        self.buf = np.zeros(n_fft, dtype=np.float32)
        self.win = np.hanning(n_fft).astype(np.float32)
        self.win_short = np.hanning(n_short).astype(np.float32)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        nyq = sr / 2 * 0.95

        # Band bin ranges (log-spaced within each zone) and per-band tilt
        self.bands: list[tuple[int, int]] = []
        centers: list[float] = []
        self.zone_bins: list[slice] = []
        for lo, hi, count in self.ZONES:
            hi = min(hi, nyq)
            edges = np.geomspace(lo, hi, count + 1)
            idx = np.searchsorted(freqs, edges)
            for a, b in zip(idx[:-1], idx[1:]):
                b = max(b, a + 1)
                self.bands.append((int(a), int(b)))
                centers.append(float(np.sqrt(freqs[a] * freqs[b - 1])) or 1.0)
            self.zone_bins.append(slice(int(idx[0]), int(idx[-1])))
        # Music falls ~3 dB/octave; tilt it flat so highs aren't always tiny
        self.tilt_db = 3.0 * np.log2(np.maximum(centers, 1.0) / 1000.0)

        # Chroma: map bins in 55 Hz - 2 kHz to pitch classes (C = 0).
        # Linear bins crowd high octaves, so weight by 1/f to keep octaves even.
        sel = np.nonzero((freqs >= 55.0) & (freqs <= 2000.0))[0]
        self.chroma_bins = sel
        midi = 69 + 12 * np.log2(freqs[sel] / 440.0)
        self.chroma_pc = np.round(midi).astype(int) % 12
        self.chroma_w = 1.0 / freqs[sel]

        # Onsets from the short window's low end (kick + bass notes)
        f_short = np.fft.rfftfreq(n_short, 1.0 / sr)
        self.kick = slice(int(np.searchsorted(f_short, 30.0)), int(np.searchsorted(f_short, 300.0)))
        self.prev_kick: np.ndarray | None = None
        self.flux_hist: deque[float] = deque(maxlen=48)  # ~1 s
        self.last_onset = -1.0
        self.clock = 0.0  # Seconds of audio analysed (sample clock)

        # Adaptive loudness references (dB), rise instantly, relax slowly
        self.ref_db = -60.0
        self.zone_ref = np.full(3, -60.0)

    def feed(self, samples: NDArray[np.float32]) -> dict:
        n = len(samples)
        self.clock += n / self.sr
        if n >= self.n:
            self.buf[:] = samples[-self.n:]
        else:
            self.buf[:-n] = self.buf[n:]
            self.buf[-n:] = samples
        rms = float(np.sqrt(np.mean(samples ** 2))) if n else 0.0
        if rms < 1e-4:
            self.prev_kick = None
            return {"type": "viz", "silent": True, "rms": 0.0}

        mag = np.abs(np.fft.rfft(self.buf * self.win)) / (self.n / 4)
        power = mag ** 2

        # Spectrum
        band_db = 10 * np.log10(np.array([power[a:b].mean() for a, b in self.bands]) + 1e-12)
        band_db += self.tilt_db
        self.ref_db = max(float(band_db.max()), self.ref_db - 0.06, -80.0)
        spec = np.clip((band_db - (self.ref_db - 48.0)) / 48.0, 0.0, 1.0)

        # Zones: each normalised to its own recent peak, so quiet highs still show movement
        zone_db = 10 * np.log10(np.array([power[z].sum() for z in self.zone_bins]) + 1e-12)
        self.zone_ref = np.maximum(zone_db, np.maximum(self.zone_ref - 0.04, -80.0))
        zones = np.clip((zone_db - (self.zone_ref - 30.0)) / 30.0, 0.0, 1.0)

        # Chroma
        chroma = np.bincount(
            self.chroma_pc, weights=power[self.chroma_bins] * self.chroma_w, minlength=12
        )
        peak = float(chroma.max())
        if peak > 0:
            chroma = chroma / peak
            note = int(chroma.argmax())
            # 1 when one note stands alone, 0 when everything is equally loud
            conf = float(1.0 - (chroma.sum() - 1.0) / 11.0)
        else:
            note, conf = 0, 0.0

        # Onset: positive spectral flux in the low end, over an adaptive threshold
        short = np.abs(np.fft.rfft(self.buf[-self.n_short:] * self.win_short))
        kick = np.log1p(short[self.kick] * 10.0)
        onset = 0.0
        if self.prev_kick is not None:
            flux = float(np.maximum(kick - self.prev_kick, 0.0).sum())
            if len(self.flux_hist) >= 8:
                h = np.array(self.flux_hist)
                thr = float(h.mean() + 1.5 * h.std())
                now = self.clock
                if flux > thr and flux > 1e-3 and now - self.last_onset > 0.18:
                    onset = float(min(1.0, (flux - thr) / (thr + 1e-6) + 0.3))
                    self.last_onset = now
            self.flux_hist.append(flux)
        self.prev_kick = kick

        r2 = lambda xs: [round(float(x), 2) for x in xs]  # noqa: E731
        return {
            "type": "viz",
            "spec": r2(spec),
            "zones": [z[2] for z in self.ZONES],
            "bands": r2(zones),
            "chroma": r2(chroma),
            "note": note,
            "conf": round(conf, 2),
            "onset": round(onset, 2),
            "rms": round(rms, 4),
        }


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
        self.viz: dict = {}
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

        v = self.viz
        if v.get("bands"):
            hue = v["note"] * 30
            name = NOTE_NAMES[v["note"]]
            grid.add_row("Note", Text(f"{name:<3}", style=f"bold {_hue_hex(hue)}") + Text(
                f" ({v['conf']:.0%} clear)", style="dim"))
            grid.add_row("Mix", Text(" ".join(
                f"{label} {_bar(level)}" for label, level in zip(("bass", "mid", "high"), v["bands"])
            ), style="dim"))

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


def _bar(level: float, width: int = 6) -> str:
    filled = level * width
    full = int(filled)
    part = " ▏▎▍▌▋▊▉"[int((filled - full) * 8)] if full < width else ""
    return ("█" * full + part).ljust(width, "·")


def _hue_hex(hue: float) -> str:
    import colorsys
    r, g, b = colorsys.hls_to_rgb((hue % 360) / 360, 0.7, 0.6)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


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
    analyzer = Analyzer(sr)

    full_url = f"{ws_url}?sr={sr}&ch=2"  # Interleaved stereo

    try:
        with Live(tui.render(), console=console, refresh_per_second=4, screen=True) as live:
            async for ws in websockets.connect(full_url):
                tui.connected = True
                live.update(tui.render())

                try:
                    async def send_audio() -> None:
                        last_render = 0.0
                        while True:
                            data = await audio_queue.get()
                            await ws.send(data)
                            mono = np.frombuffer(data, dtype=np.float32).reshape(-1, 2).mean(axis=1)
                            frame = analyzer.feed(mono.astype(np.float32))
                            await ws.send(json.dumps(frame, separators=(",", ":")))
                            now = time.monotonic()
                            if now - last_render > 0.25:
                                last_render = now
                                tui.viz = frame
                                live.update(tui.render())

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
