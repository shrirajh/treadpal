"""Energy-based intensity tracking for incline control."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import librosa  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from treadpal.app import AppState

logger = logging.getLogger("treadpal.audio.intensity")

# Module-level state for rolling baseline
_energy_history: list[float] = []
_MAX_HISTORY = 30


def compute_intensity_incline(
    audio: NDArray[np.float32],
    sr: int,
    state: AppState,
) -> float | None:
    """Compute incline from audio energy + spectral brightness.

    Returns target incline based on home_incline ± incline_range from config,
    or None if intensity tracking is not configured.
    """
    cfg = state.config
    home_incline = getattr(cfg, "home_incline", None)
    if home_incline is None:
        return None
    incline_range = getattr(cfg, "incline_range", 4.0)

    rms = float(np.sqrt(np.mean(audio ** 2)))
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    avg_centroid = float(np.mean(centroid))
    norm_centroid = min(1.0, max(0.0, (avg_centroid - 500) / 4500))

    intensity = 0.6 * min(1.0, rms / 0.15) + 0.4 * norm_centroid

    _energy_history.append(intensity)
    if len(_energy_history) > _MAX_HISTORY:
        _energy_history.pop(0)

    if len(_energy_history) < 3:
        return home_incline

    baseline = float(np.mean(_energy_history))
    std = float(np.std(_energy_history))
    if std < 0.01:
        return home_incline

    z_score = (intensity - baseline) / std
    z_clamped = max(-2.0, min(2.0, z_score))
    offset = (z_clamped / 2.0) * incline_range

    return round(max(0.0, min(15.0, home_incline + offset)), 1)
