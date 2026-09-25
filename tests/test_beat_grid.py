"""Tests for beat grid fitting (beat phase)."""

from __future__ import annotations

import pytest

from treadpal.audio.beat_grid import fit_beat_grid


def test_grid_finds_last_beat() -> None:
    # 120 BPM, beats at 0.1, 0.6, ... within an 8 s window
    beats = [0.1 + 0.5 * i for i in range(16)]
    grid = fit_beat_grid(beats, [], 0.5, 8.0)
    assert grid is not None
    assert grid.last_beat_s == pytest.approx(7.6, abs=1e-6)
    assert grid.confidence == pytest.approx(1.0, abs=1e-3)


def test_grid_tolerates_jitter_and_outlier() -> None:
    beats = [0.1 + 0.5 * i + (0.01 if i % 2 else -0.01) for i in range(16)]
    beats[5] += 0.2  # One badly placed detection
    grid = fit_beat_grid(beats, [], 0.5, 8.0)
    assert grid is not None
    assert grid.last_beat_s == pytest.approx(7.6, abs=0.03)
    assert 0.5 < grid.confidence < 1.0


def test_grid_wraps_phase_near_zero() -> None:
    # Beats straddle the period boundary (0.49 / 0.01 mod 0.5): circular mean must not average to 0.25
    beats = [0.49 + 0.5 * i + (0.02 if i % 2 else 0.0) for i in range(10)]
    grid = fit_beat_grid(beats, [], 0.5, 5.3)
    assert grid is not None
    assert grid.last_beat_s % 0.5 == pytest.approx(0.0, abs=0.02) or grid.last_beat_s % 0.5 == pytest.approx(0.5, abs=0.02)


def test_grid_bar_from_downbeats() -> None:
    beats = [0.25 + 0.5 * i for i in range(16)]
    downbeats = [0.25, 2.25, 4.25, 6.25]
    grid = fit_beat_grid(beats, downbeats, 0.5, 8.0)
    assert grid is not None
    assert grid.bar_length == 4
    assert grid.last_downbeat_s == pytest.approx(6.25, abs=1e-6)


def test_folded_period_still_on_beats() -> None:
    # 70 BPM track folded to 140: every real beat is on the doubled grid
    period = 60 / 70
    beats = [0.3 + period * i for i in range(9)]
    grid = fit_beat_grid(beats, [], period / 2, 8.0)
    assert grid is not None
    assert ((grid.last_beat_s - 0.3) / (period / 2)) == pytest.approx(
        round((grid.last_beat_s - 0.3) / (period / 2)), abs=1e-3
    )


def test_too_few_beats() -> None:
    assert fit_beat_grid([1.0], [], 0.5, 8.0) is None
