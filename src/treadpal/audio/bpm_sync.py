"""BPM-to-speed harmonic snapping with biomechanical stride model.

For each BPM harmonic candidate, scans the user's speed range and scores
each (harmonic, speed) pair by how closely the implied stride matches the
biomechanically natural stride at that speed. The best-scoring pair wins.

Supports manual harmonic override (gear up/down) that auto-releases on song change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def natural_stride(speed_kmh: float) -> float:
    """Expected stride length (m) at a given speed, from biomechanics data.

    Linear approximation:
      Walking (< 7 km/h):  stride = 0.35 + 0.075 * speed
      Running (>= 7 km/h): stride = 0.55 + 0.065 * speed
    """
    if speed_kmh < 7.0:
        return 0.35 + 0.075 * speed_kmh
    return 0.55 + 0.065 * speed_kmh


def _gaussian(x: float, center: float, sigma: float) -> float:
    return math.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def _is_same_song(bpm_a: float, bpm_b: float, tolerance: float = 10.0) -> bool:
    """Check if two BPMs are from the same song (within tolerance, octave-aware)."""
    for mult in (0.5, 1.0, 2.0):
        if abs(bpm_a * mult - bpm_b) <= tolerance:
            return True
    return False


@dataclass
class BpmSyncResult:
    detected_bpm: float
    selected_harmonic: float
    effective_cadence: float
    implied_stride_m: float
    natural_stride_m: float
    stride_score: float
    speed_kmh: float
    harmonic_override: bool


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
        # Harmonic override
        self._forced_harmonic: float | None = None
        self._forced_bpm: float | None = None

    @property
    def forced_harmonic(self) -> float | None:
        return self._forced_harmonic

    def shift_harmonic(self, direction: int) -> float | None:
        """Shift to next/previous harmonic. direction: +1 = up, -1 = down.

        Returns the new forced harmonic, or None if at the limit.
        """
        sorted_h = sorted(self.harmonics)
        last = self._last_result
        if last is None:
            return None

        current = self._forced_harmonic or last.selected_harmonic
        try:
            idx = sorted_h.index(current)
        except ValueError:
            # Current harmonic not in list, find nearest
            idx = min(range(len(sorted_h)), key=lambda i: abs(sorted_h[i] - current))

        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(sorted_h):
            return self._forced_harmonic  # At limit

        self._forced_harmonic = sorted_h[new_idx]
        self._forced_bpm = last.detected_bpm
        return self._forced_harmonic

    def reset_harmonic(self) -> None:
        """Clear manual harmonic override, return to auto."""
        self._forced_harmonic = None
        self._forced_bpm = None

    def _compute_for_harmonic(self, bpm: float, h: float) -> tuple[float, float, float, float, float]:
        """Compute best speed for a specific harmonic. Returns (score, speed, cadence, implied_stride, natural_stride)."""
        cadence = bpm * h
        best_score = -1.0
        best_speed = (self.min_speed_kmh + self.max_speed_kmh) / 2
        best_implied = 0.7
        best_natural = 0.7

        speed = self.min_speed_kmh
        while speed <= self.max_speed_kmh + 1e-9:
            implied_stride = (speed * 1000.0) / (cadence * 60.0)
            expected_stride = natural_stride(speed)
            score = _gaussian(implied_stride, expected_stride, self.stride_sigma)
            if score > best_score:
                best_score = score
                best_speed = speed
                best_implied = implied_stride
                best_natural = expected_stride
            speed = round(speed + self.speed_step_kmh, 2)

        return best_score, best_speed, cadence, best_implied, best_natural

    def compute(self, bpm: float) -> BpmSyncResult:
        """Find the (harmonic, speed) pair with the most natural stride fit."""
        override = False

        # Check if forced harmonic should be released (song changed)
        if self._forced_harmonic is not None and self._forced_bpm is not None:
            if _is_same_song(bpm, self._forced_bpm):
                override = True
            else:
                self._forced_harmonic = None
                self._forced_bpm = None

        if override and self._forced_harmonic is not None:
            h = self._forced_harmonic
            cadence = bpm * h
            if 30 <= cadence <= 250:
                score, speed, cadence, implied, natural_s = self._compute_for_harmonic(bpm, h)
                result = BpmSyncResult(
                    detected_bpm=bpm,
                    selected_harmonic=h,
                    effective_cadence=round(cadence, 1),
                    implied_stride_m=round(implied, 3),
                    natural_stride_m=round(natural_s, 3),
                    stride_score=round(score, 3),
                    speed_kmh=round(speed, 2),
                    harmonic_override=True,
                )
                self._last_result = result
                return result
            # Forced harmonic produces impossible cadence, fall through to auto
            self._forced_harmonic = None
            self._forced_bpm = None

        # Auto mode: scan all harmonics
        best_score = -1.0
        best_speed = (self.min_speed_kmh + self.max_speed_kmh) / 2
        best_harmonic = 1.0
        best_cadence = bpm
        best_implied_stride = 0.7
        best_natural_stride = 0.7

        for h in self.harmonics:
            cadence = bpm * h
            if cadence < 30 or cadence > 250:
                continue

            score, speed, cadence, implied, natural_s = self._compute_for_harmonic(bpm, h)
            if score > best_score:
                best_score = score
                best_speed = speed
                best_harmonic = h
                best_cadence = cadence
                best_implied_stride = implied
                best_natural_stride = natural_s

        result = BpmSyncResult(
            detected_bpm=bpm,
            selected_harmonic=best_harmonic,
            effective_cadence=round(best_cadence, 1),
            implied_stride_m=round(best_implied_stride, 3),
            natural_stride_m=round(best_natural_stride, 3),
            stride_score=round(best_score, 3),
            speed_kmh=round(best_speed, 2),
            harmonic_override=False,
        )
        self._last_result = result
        return result

    @property
    def last_result(self) -> BpmSyncResult | None:
        return self._last_result
