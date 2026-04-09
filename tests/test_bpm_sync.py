"""Tests for BPM harmonic snapping with biomechanical stride model."""

from __future__ import annotations

from treadpal.audio.bpm_sync import BpmSyncController, natural_stride


class TestNaturalStride:
    def test_walking_speeds(self) -> None:
        assert 0.55 < natural_stride(3.0) < 0.65
        assert 0.65 < natural_stride(5.0) < 0.80
        assert 0.75 < natural_stride(6.0) < 0.85

    def test_running_speeds(self) -> None:
        assert 0.95 < natural_stride(8.0) < 1.15
        assert 1.10 < natural_stride(10.0) < 1.30

    def test_monotonically_increasing(self) -> None:
        speeds = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
        strides = [natural_stride(s) for s in speeds]
        for i in range(len(strides) - 1):
            assert strides[i] < strides[i + 1]


class TestBpmSyncController:
    def _make(self, **kwargs: object) -> BpmSyncController:
        defaults = {"min_speed_kmh": 4.0, "max_speed_kmh": 7.0, "speed_step_kmh": 0.1}
        defaults.update(kwargs)
        return BpmSyncController(**defaults)  # type: ignore[arg-type]

    def test_result_within_speed_range(self) -> None:
        ctrl = self._make(min_speed_kmh=4.0, max_speed_kmh=7.0)
        for bpm in [80, 100, 120, 140, 160, 180]:
            result = ctrl.compute(bpm)
            assert 4.0 <= result.speed_kmh <= 7.0, f"BPM={bpm}: speed={result.speed_kmh}"

    def test_85_and_170_similar_speed(self) -> None:
        """85 and 170 BPM should produce similar speeds (octave equivalence)."""
        ctrl = self._make()
        r85 = ctrl.compute(85)
        r170 = ctrl.compute(170)
        # They might not be identical but should be within the same range
        assert abs(r85.speed_kmh - r170.speed_kmh) < 2.0

    def test_stride_score_positive(self) -> None:
        ctrl = self._make()
        result = ctrl.compute(120)
        assert result.stride_score > 0

    def test_implied_stride_reasonable(self) -> None:
        ctrl = self._make(min_speed_kmh=4.0, max_speed_kmh=7.0)
        result = ctrl.compute(130)
        # At walking speeds, stride should be 0.4-1.0m
        assert 0.3 < result.implied_stride_m < 1.2

    def test_natural_stride_close_to_implied(self) -> None:
        """The algorithm should pick pairs where implied ≈ natural stride."""
        ctrl = self._make()
        result = ctrl.compute(130)
        # The whole point: implied should be close to natural
        assert abs(result.implied_stride_m - result.natural_stride_m) < 0.2

    def test_last_result(self) -> None:
        ctrl = self._make()
        assert ctrl.last_result is None
        ctrl.compute(120)
        assert ctrl.last_result is not None
        assert ctrl.last_result.detected_bpm == 120

    def test_different_speed_ranges(self) -> None:
        """Higher speed range should produce higher speeds."""
        slow = BpmSyncController(min_speed_kmh=3.0, max_speed_kmh=5.0)
        fast = BpmSyncController(min_speed_kmh=8.0, max_speed_kmh=12.0)
        r_slow = slow.compute(140)
        r_fast = fast.compute(140)
        assert r_slow.speed_kmh < r_fast.speed_kmh

    def test_result_fields_populated(self) -> None:
        ctrl = self._make()
        result = ctrl.compute(128)
        assert result.detected_bpm == 128
        assert result.selected_harmonic in ctrl.harmonics
        assert result.effective_cadence > 0
        assert result.implied_stride_m > 0
        assert result.natural_stride_m > 0
        assert 0 <= result.stride_score <= 1.0
