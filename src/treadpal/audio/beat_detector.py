"""Server-side BPM detection using beat_this (ISMIR 2024 SOTA).

Manages a rolling audio buffer fed by WebSocket clients, runs beat_this
periodically, and returns BPM computed from inter-beat intervals.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import torch
from numpy.typing import NDArray

from treadpal.audio.beat_grid import BeatGrid, fit_beat_grid

logger = logging.getLogger("treadpal.beat")

_model: object = None
_model_lock = threading.Lock()


def _get_model() -> object:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from beat_this.inference import Audio2Beats
                device = "cuda" if torch.cuda.is_available() else "cpu"
                # More threads barely speed up an 8 s window (4: ~150 ms, 16: ~80 ms)
                # but take cores from the stem separator and everything else
                torch.set_num_threads(min(4, torch.get_num_threads()))
                _model = Audio2Beats(checkpoint_path="final0", device=device)
                logger.info("beat_this model loaded on %s", device)
    return _model


def preload() -> None:
    """Load the model ahead of the first detection."""
    _get_model()


class AudioBuffer:
    """Thread-safe circular buffer of mono float32 audio."""

    def __init__(self, sr: int, max_seconds: float = 12.0) -> None:
        self.sr = sr
        self._max_samples = int(sr * max_seconds)
        self._buf = np.zeros(self._max_samples, dtype=np.float32)
        self._write_pos = 0
        self._total_written = 0
        self._lock = threading.Lock()

    def append(self, data: NDArray[np.float32]) -> None:
        with self._lock:
            n = len(data)
            if n >= self._max_samples:
                self._buf[:] = data[-self._max_samples:]
                self._write_pos = 0
                self._total_written += n
                return
            end = self._write_pos + n
            if end <= self._max_samples:
                self._buf[self._write_pos:end] = data
            else:
                first = self._max_samples - self._write_pos
                self._buf[self._write_pos:] = data[:first]
                self._buf[:n - first] = data[first:]
            self._write_pos = end % self._max_samples
            self._total_written += n

    def get_last(self, seconds: float) -> NDArray[np.float32] | None:
        n = int(self.sr * seconds)
        with self._lock:
            if self._total_written < n:
                return None
            end = self._write_pos
            start = end - n
            if start >= 0:
                return self._buf[start:end].copy()
            else:
                return np.concatenate([self._buf[start:], self._buf[:end]]).copy()

    @property
    def seconds_available(self) -> float:
        return min(self._total_written, self._max_samples) / self.sr


def detect_beats(audio: NDArray[np.float32], sr: int) -> tuple[float, BeatGrid | None] | None:
    """Detect BPM and the beat grid using beat_this.

    Returns (bpm, grid) or None if detection fails. BPM is folded into
    60-180; the grid uses the folded period and positions are seconds from
    the start of ``audio``.
    """
    model = _get_model()
    tensor = torch.from_numpy(audio)

    logger.debug("Running beat_this on %.1fs audio (sr=%d, samples=%d, range=[%.4f, %.4f])",
                 len(audio) / sr, sr, len(audio), float(audio.min()), float(audio.max()))

    beats, downbeats = model(tensor, sr)  # ty: ignore[call-non-callable]

    logger.debug("beat_this returned %d beats, %d downbeats", len(beats), len(downbeats))

    if len(beats) < 3:
        logger.debug("Too few beats (%d) for BPM calculation", len(beats))
        return None
    ibis = np.diff(beats)
    ibis = ibis[(ibis > 0.2) & (ibis < 2.0)]
    if len(ibis) < 2:
        logger.debug("Not enough valid IBIs after filtering (%d)", len(ibis))
        return None

    bpm = 60.0 / float(np.median(ibis))
    while bpm < 60:
        bpm *= 2
    while bpm > 180:
        bpm /= 2
    grid = fit_beat_grid(beats, downbeats, 60.0 / bpm, len(audio) / sr)
    logger.debug(
        "BPM=%.1f from %d valid IBIs (median=%.3fs), grid=%s",
        bpm, len(ibis), float(np.median(ibis)), grid,
    )
    return round(bpm, 1), grid
