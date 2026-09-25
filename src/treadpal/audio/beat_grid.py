"""Beat phase: fit a regular grid to detected beat times.

beat_this gives beat positions within an analysis window. BPM alone can't
place a beat in time; the grid does. Phase is a circular mean of beat times
modulo the period, weighted toward recent beats (the ones that predict what
comes next), so a few misplaced detections don't drag it around.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class BeatGrid:
    period_s: float
    last_beat_s: float  # Grid beat position within the window (seconds from window start)
    confidence: float  # 0..1: how tightly detected beats sit on the grid
    bar_length: int | None = None  # Beats per bar, if downbeats were found
    last_downbeat_s: float | None = None  # Grid-snapped downbeat position within the window


def fit_beat_grid(
    beats: NDArray[np.floating] | list[float],
    downbeats: NDArray[np.floating] | list[float],
    period_s: float,
    window_s: float,
) -> BeatGrid | None:
    """Fit beats to t = t0 + n * period and return the last grid beat before window end.

    ``period_s`` is the (possibly octave-folded) beat period used for display; a
    folded grid still lands on real beats (every beat, or every other one).
    """
    b = np.asarray(beats, dtype=np.float64)
    if len(b) < 2 or period_s <= 0:
        return None

    angles = 2 * math.pi * (b % period_s) / period_s
    # Recent beats matter most: linear ramp from 0.3 (oldest) to 1.0 (newest)
    weights = 0.3 + 0.7 * (b - b.min()) / max(float(np.ptp(b)), 1e-9)
    c = float(np.sum(weights * np.cos(angles)))
    s = float(np.sum(weights * np.sin(angles)))
    confidence = math.hypot(c, s) / float(np.sum(weights))
    t0 = (math.atan2(s, c) % (2 * math.pi)) / (2 * math.pi) * period_s

    n_last = math.floor((window_s - t0) / period_s)
    last_beat = t0 + n_last * period_s

    grid = BeatGrid(period_s=period_s, last_beat_s=last_beat, confidence=round(confidence, 3))

    d = np.asarray(downbeats, dtype=np.float64)
    if len(d) >= 2:
        bar = int(round(float(np.median(np.diff(d))) / period_s))
        if 2 <= bar <= 8:
            grid.bar_length = bar
            # Snap the latest downbeat onto the beat grid
            n_down = round((float(d[-1]) - t0) / period_s)
            grid.last_downbeat_s = t0 + n_down * period_s
    return grid
