"""BPM-to-speed harmonic snapping with biomechanical stride model.

For each BPM harmonic candidate, scans the user's speed range and scores
each (harmonic, speed) pair by how closely the implied stride matches the
biomechanically natural stride at that speed. The best-scoring pair wins.

User only needs to set min/max speed — no stride length input required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def natural_stride(speed_kmh: float) -> float:
    """Expected stride length (m) at a given speed, from biomechanics data.

    Linear approximation:
      Walking (< 7 km/h):  stride ≈ 0.35 + 0.075 × speed
      Running (≥ 7 km/h):  stride ≈ 0.55 + 0.065 × speed
    """
    if speed_kmh < 7.0:
        return 0.35 + 0.075 * speed_kmh
    return 0.55 + 0.065 * speed_kmh


def _gaussian(x: float, center: float, sigma: float) -> float:
    return math.exp(-((x - center) ** 2) / (2 * sigma ** 2))


@dataclass
class BpmSyncResult:
    detected_bpm: float
    selected_harmonic: float
    effective_cadence: float
    implied_stride_m: float
    natural_stride_m: float
    stride_score: float
    speed_kmh: float


class BpmSyncController:
    """Harmonic snapping BPM-to-speed controller using biomechanical stride model."""

    def __init__(
        self,
        min_speed_kmh: float = 4.0,
        max_speed_kmh: float = 7.0,
        harmonics: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0),
        speed_step_kmh: float = 0.1,
        stride_sigma: float = 0.08,
    ) -> None:
        self.min_speed_kmh = min_speed_kmh
        self.max_speed_kmh = max_speed_kmh
        self.harmonics = harmonics
        self.speed_step_kmh = speed_step_kmh
        self.stride_sigma = stride_sigma
        self._last_result: BpmSyncResult | None = None

    def compute(self, bpm: float) -> BpmSyncResult:
        """Find the (harmonic, speed) pair with the most natural stride fit."""
        best_score = -1.0
        best_speed = (self.min_speed_kmh + self.max_speed_kmh) / 2
        best_harmonic = 1.0
        best_cadence = bpm
        best_implied_stride = 0.7
        best_natural_stride = 0.7

        # Scan all (harmonic, speed) combinations
        for h in self.harmonics:
            cadence = bpm * h
            if cadence < 30 or cadence > 250:
                continue  # Physically impossible cadences

            speed = self.min_speed_kmh
            while speed <= self.max_speed_kmh + 1e-9:
                # What stride would this (cadence, speed) pair imply?
                implied_stride = (speed * 1000.0) / (cadence * 60.0)
                # What stride is natural at this speed?
                expected_stride = natural_stride(speed)
                # How well do they match?
                score = _gaussian(implied_stride, expected_stride, self.stride_sigma)

                if score > best_score:
                    best_score = score
                    best_speed = speed
                    best_harmonic = h
                    best_cadence = cadence
                    best_implied_stride = implied_stride
                    best_natural_stride = expected_stride

                speed = round(speed + self.speed_step_kmh, 2)

        result = BpmSyncResult(
            detected_bpm=bpm,
            selected_harmonic=best_harmonic,
            effective_cadence=round(best_cadence, 1),
            implied_stride_m=round(best_implied_stride, 3),
            natural_stride_m=round(best_natural_stride, 3),
            stride_score=round(best_score, 3),
            speed_kmh=round(best_speed, 2),
        )
        self._last_result = result
        return result

    @property
    def last_result(self) -> BpmSyncResult | None:
        return self._last_result
