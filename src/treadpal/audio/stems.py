"""Real-time stem separation (vocals / other / bass / drums) for the visualiser.

Runs the pretrained streaming HS-TasNet released with StemgenRT, vendored in
treadpal/vendor/hs_tasnet (see its README for provenance, licences and how
it was rebuilt): 44.1 kHz stereo in 128-sample hops, one hop (2.9 ms) of
latency, recurrent state carried between hops.

The vendored graph has its Conv/DFT ops rewritten as matrix products and its
weights quantized to int8, which cuts CPU time per hop from ~4 ms (140% of
real time on one core) to ~1 ms. Stem levels stay within ~0.5 dB of the
float32 model.
"""

from __future__ import annotations

import hashlib
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("treadpal.audio.stems")

MODEL_PATH = Path(__file__).resolve().parents[1] / "vendor" / "hs_tasnet" / "hop128-int8.onnx"
MODEL_SHA256 = "3bfccc29554bca9446fdaa5ff6b5132f88a14ab4ac685f677a8e64ecc4667a67"
SR = 44100
HOP = 128
MODEL_ORDER = ("drums", "bass", "vocals", "other")
DISPLAY_ORDER = ("vocals", "other", "bass", "drums")  # Top to bottom, like a score
_DISPLAY_IDX = [MODEL_ORDER.index(n) for n in DISPLAY_ORDER]
_STATES = {
    "audio_history": (1, 2, 896),
    "fusion_hidden": (2, 1, 1000),
    "spectral_numerator_tail": (1, 4, 2, 128),
    "waveform_tail": (1, 4, 2, 128),
}


def verified_model(path: Path = MODEL_PATH) -> Path:
    """The vendored model, if intact (not, say, a Git LFS pointer or a partial checkout)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != MODEL_SHA256:
        raise ValueError(f"Stem model {path} doesn't match its SHA-256")
    return path


class StemModel:
    """The loaded model, shared by all streams (their recurrent state is passed in per call).

    Built once at startup and warmed up, so a new stream doesn't stall the first
    seconds of audio while ONNX Runtime allocates and settles.
    """

    def __init__(self, model_path: Path = MODEL_PATH, threads: int = 2) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.log_severity_level = 3
        # Don't burn idle cores spinning between hops
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        inputs = {n.name: tuple(n.shape) for n in self.sess.get_inputs()}
        if inputs != {"audio_chunk": (1, 2, HOP), **_STATES}:
            raise ValueError(f"Unexpected stem model interface: {inputs}")
        self.outputs = ["separated_chunk"] + ["next_" + k for k in _STATES]
        warm = StemSession(self)
        noise = np.random.default_rng(0).standard_normal((2, HOP)).astype(np.float32) * 0.1
        for _ in range(200):
            warm.process(noise)


class StemSession:
    """One stream through the model: 128-sample stereo hops in, 4 stems out."""

    def __init__(self, model: StemModel) -> None:
        self.model = model
        self.reset()

    def reset(self) -> None:
        self._state = {k: np.zeros(s, dtype=np.float32) for k, s in _STATES.items()}
        self._primed = False

    def process(self, hop: np.ndarray) -> np.ndarray | None:
        """[2, 128] float32 -> [4, 2, 128] for the previous hop (None right after reset)."""
        out = self.model.sess.run(self.model.outputs, {"audio_chunk": hop[None], **self._state})
        self._state = dict(zip(_STATES, out[1:]))
        primed, self._primed = self._primed, True
        return out[0][0] if primed else None


class StemFeatures:
    """Per-stem level, spectrum and chroma (so each stem has its own note), plus drum onsets.

    All stems share one dB scale, anchored to the whole song's recent peak
    loudness: the top of a lane is that peak, and the bottom is `floor_db`
    below it (and never below an absolute floor, so near-silence isn't blown
    up). A faint sound stays small, and model bleed from an absent stem (often
    30-40 dB down) shows nothing. The scale follows the music's loudness
    slowly, so it doesn't pump with every kick.
    """

    ABS_FLOOR_DB = -70.0  # dBFS: never show anything quieter than this
    REF_DECAY_DB = 0.02  # Per frame (~0.9 dB/s): how fast the scale follows a quieter song

    FRAME = 1024  # Samples per output frame (~43 fps)
    N_SPEC = 16
    N_LONG = 8192  # Chroma needs bass-note resolution (~5 Hz)
    N_SHORT = 2048

    def __init__(self, floor_db: float = 36.0) -> None:
        self.floor_db = floor_db
        self.buf = np.zeros((4, self.N_LONG), dtype=np.float32)  # Mono per stem, model order
        self.win = np.hanning(self.N_SHORT).astype(np.float32)
        self.win_long = np.hanning(self.N_LONG).astype(np.float32)
        f = np.fft.rfftfreq(self.N_SHORT, 1 / SR)
        edges = np.searchsorted(f, np.geomspace(40.0, 16000.0, self.N_SPEC + 1))
        self.bands = [(int(a), int(max(b, a + 1))) for a, b in zip(edges[:-1], edges[1:])]
        centers = np.sqrt(np.geomspace(40.0, 16000.0, self.N_SPEC + 1)[:-1] * np.geomspace(40.0, 16000.0, self.N_SPEC + 1)[1:])
        self.tilt_db = 3.0 * np.log2(centers / 1000.0)  # Flatten music's ~3 dB/octave fall
        self.kick = slice(int(np.searchsorted(f, 30.0)), int(np.searchsorted(f, 300.0)))
        fl = np.fft.rfftfreq(self.N_LONG, 1 / SR)
        self.chroma_bins = np.nonzero((fl >= 30.0) & (fl <= 2000.0))[0]
        f_c = fl[self.chroma_bins]
        self.chroma_pc = np.round(69 + 12 * np.log2(f_c / 440.0)).astype(int) % 12
        # Per stem: the range its notes (and their first harmonics) live in, weighted
        # 1/f so every octave counts about equally despite linear FFT bins
        ranges = {"drums": (40.0, 2000.0), "bass": (30.0, 1000.0), "vocals": (80.0, 2000.0), "other": (55.0, 2000.0)}
        self.chroma_w = np.stack([
            np.where((f_c >= ranges[n][0]) & (f_c <= ranges[n][1]), 1.0 / f_c, 0.0) for n in MODEL_ORDER
        ])
        self.level_ref = -80.0  # Recent peak loudness of the whole song (dBFS)
        self.spec_ref = -80.0  # Recent peak band level over all stems
        self.prev_kick: np.ndarray | None = None
        self.flux_hist: deque[float] = deque(maxlen=48)
        self.since_onset = 1.0
        self.pending = 0

    def feed(self, stems: np.ndarray) -> list[dict[str, Any]]:
        """Add [4, 2, n] stem audio; return any completed frames."""
        mono = stems.mean(axis=1)
        frames = []
        pos = 0
        n = mono.shape[1]
        while pos < n:
            take = min(n - pos, self.FRAME - self.pending)
            chunk = mono[:, pos:pos + take]
            self.buf = np.roll(self.buf, -take, axis=1)
            self.buf[:, -take:] = chunk
            self.pending += take
            pos += take
            if self.pending == self.FRAME:
                self.pending = 0
                frames.append(self._frame())
        return frames

    def _scale(self, db: np.ndarray, ref: float) -> np.ndarray:
        """0 at the floor, 1 at the song's recent peak."""
        lo = max(ref - self.floor_db, self.ABS_FLOOR_DB)
        return np.clip((db - lo) / max(ref - lo, 6.0), 0.0, 1.0)

    def _frame(self) -> dict[str, Any]:
        recent = self.buf[:, -self.FRAME:]
        self.since_onset += self.FRAME / SR
        if float(np.sqrt(np.mean(recent.sum(axis=0) ** 2))) < 1e-4:
            self.prev_kick = None
            return {"type": "stems", "silent": True}

        # Levels, on one scale: the mix's recent peak at the top, floor_db below it at the bottom
        db = 10 * np.log10(np.mean(recent ** 2, axis=1) + 1e-12)
        mix_db = 10 * np.log10(np.mean(recent.sum(axis=0) ** 2) + 1e-12)
        self.level_ref = max(mix_db, self.level_ref - self.REF_DECAY_DB)
        level = self._scale(db, self.level_ref)

        # Spectra
        short = self.buf[:, -self.N_SHORT:] * self.win
        mag = np.abs(np.fft.rfft(short, axis=1)) / (self.N_SHORT / 4)
        power = mag ** 2
        band_db = 10 * np.log10(np.stack([power[:, a:b].mean(axis=1) for a, b in self.bands], axis=1) + 1e-12)
        band_db += self.tilt_db
        self.spec_ref = max(float(band_db.max()), self.spec_ref - self.REF_DECAY_DB)
        spec = self._scale(band_db, self.spec_ref)

        # Chroma per stem (silent below the floor, so bleed has no note), and overall
        # from the pitched stems only: drums would smear every pitch class
        p = np.abs(np.fft.rfft(self.buf * self.win_long, axis=1)[:, self.chroma_bins]) ** 2 * self.chroma_w
        raw = np.stack([np.bincount(self.chroma_pc, weights=p[i], minlength=12) for i in range(4)])
        raw[level == 0] = 0.0
        pitched = [MODEL_ORDER.index(n) for n in ("bass", "other", "vocals")]
        chroma = raw[pitched].sum(axis=0)
        chroma = chroma / chroma.max() if chroma.max() > 0 else chroma
        peak = raw.max(axis=1, keepdims=True)
        stem_chroma = np.divide(raw, peak, out=np.zeros_like(raw), where=peak > 0)

        # Kick onsets from the drum stem's low end: no bass notes to confuse it
        kick = np.log1p(mag[MODEL_ORDER.index("drums"), self.kick] * 10.0)
        onset = 0.0
        if self.prev_kick is not None:
            flux = float(np.maximum(kick - self.prev_kick, 0.0).sum())
            if len(self.flux_hist) >= 8:
                h = np.array(self.flux_hist)
                thr = float(h.mean() + 1.5 * h.std())
                if flux > thr and flux > 1e-3 and self.since_onset > 0.18:
                    onset = float(min(1.0, (flux - thr) / (thr + 1e-6) + 0.3))
                    self.since_onset = 0.0
            self.flux_hist.append(flux)
        self.prev_kick = kick

        r2 = lambda xs: [round(float(x), 2) for x in xs]  # noqa: E731
        return {
            "type": "stems",
            "names": list(DISPLAY_ORDER),
            "level": r2(level[_DISPLAY_IDX]),
            "spec": r2(spec[_DISPLAY_IDX].ravel()),
            "bands": self.N_SPEC,
            "chroma": r2(chroma),
            "stem_chroma": r2(stem_chroma[_DISPLAY_IDX].ravel()),  # 12 per stem, in `names` order
            "onset": round(onset, 2),
        }


class StemWorker:
    """Separates one audio connection on its own thread and emits stem frames.

    Audio in at any rate/channel count; resampled to 44.1 kHz stereo. If the
    CPU can't keep up the backlog is dropped (and state reset) rather than
    letting the visuals lag behind the music.
    """

    MAX_BACKLOG_S = 0.5  # At ~40% load, a full backlog drains in ~0.3 s

    def __init__(
        self, model: StemModel, sr: int, emit: Callable[[dict[str, Any]], None], floor_db: float = 36.0
    ) -> None:
        self.sr = sr
        self.emit = emit
        self.session = StemSession(model)
        self.features = StemFeatures(floor_db)
        self._q: queue.Queue[tuple[np.ndarray, float] | None] = queue.Queue()
        self._backlog = 0  # Input samples queued
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="stem-separator", daemon=True)
        self.busy_s = 0.0  # Processing time, for load reporting
        self.audio_s = 0.0
        self.dropped = 0
        self._thread.start()

    def feed(self, stereo: np.ndarray) -> None:
        """Queue [2, n] float32 audio that just arrived."""
        with self._lock:
            self._backlog += stereo.shape[1]
        self._q.put((stereo, time.monotonic()))

    def close(self) -> None:
        self._q.put(None)

    def _run(self) -> None:
        import soxr

        resampler = soxr.ResampleStream(self.sr, SR, 2, dtype="float32") if self.sr != SR else None
        pending = np.zeros((2, 0), dtype=np.float32)
        silent_hops = 0
        while True:
            item = self._q.get()
            if item is None:
                return
            stereo, arrived = item
            with self._lock:
                self._backlog -= stereo.shape[1]
                behind = self._backlog / self.sr
            if behind > self.MAX_BACKLOG_S:
                # Skip ahead: drop what's queued and restart the stream cleanly
                dropped = 0
                while True:
                    try:
                        nxt = self._q.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None:
                        return
                    stereo, arrived = nxt
                    with self._lock:
                        self._backlog -= stereo.shape[1]
                    dropped += 1
                self.dropped += dropped
                logger.warning("Stem separation fell %.2fs behind; skipped %d blocks", behind, dropped)
                self.session.reset()
                pending = pending[:, :0]

            t0 = time.perf_counter()
            x = stereo if resampler is None else resampler.resample_chunk(np.ascontiguousarray(stereo.T)).T
            pending = np.concatenate([pending, x.astype(np.float32, copy=False)], axis=1)
            outs = []
            n_hops = pending.shape[1] // HOP
            for i in range(n_hops):
                hop = np.ascontiguousarray(pending[:, i * HOP:(i + 1) * HOP])
                if not np.any(np.abs(hop) > 1e-5):
                    # Silence: no point separating it. Restart cleanly when sound returns.
                    silent_hops += 1
                    if silent_hops == 1:
                        self.session.reset()
                    outs.append(np.zeros((4, 2, HOP), dtype=np.float32))
                    continue
                silent_hops = 0
                y = self.session.process(hop)
                if y is not None:
                    outs.append(y)
            pending = pending[:, n_hops * HOP:]
            self.busy_s += time.perf_counter() - t0
            self.audio_s += n_hops * HOP / SR
            if outs:
                for frame in self.features.feed(np.concatenate(outs, axis=-1)):
                    # How long after the audio arrived this frame is ready, so the UI
                    # can place onsets where they happened
                    frame["lag"] = round(time.monotonic() - arrived, 4)
                    self.emit(frame)

    @property
    def load(self) -> float:
        """Fraction of real time spent separating (1.0 = can't keep up)."""
        return self.busy_s / self.audio_s if self.audio_s else 0.0
