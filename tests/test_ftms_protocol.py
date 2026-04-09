"""Tests for FTMS binary protocol parsing."""

from __future__ import annotations

import struct

from treadpal.ble.ftms_protocol import (
    parse_features,
    parse_heart_rate,
    parse_machine_status,
    parse_treadmill_data,
)


class TestParseTreadmillData:
    def test_speed_only(self) -> None:
        """Flags 0x0000: bit 0 = 0 means speed IS present, nothing else."""
        # 6.00 km/h = 600 in 0.01 units
        data = struct.pack("<HH", 0x0000, 600)
        result = parse_treadmill_data(data)
        assert result.speed_kmh == 6.0
        assert result.incline_pct == 0.0
        assert result.distance_m == 0
        assert result.elapsed_time_s == 0
        assert result.calories_kcal == 0
        assert result.heart_rate_bpm is None

    def test_speed_not_present(self) -> None:
        """Flags bit 0 = 1 means speed NOT present (inverted)."""
        data = struct.pack("<H", 0x0001)
        result = parse_treadmill_data(data)
        assert result.speed_kmh == 0.0

    def test_speed_and_distance(self) -> None:
        """Flags: speed present + total distance (bit 2)."""
        flags = 0x0004  # bit 2 set
        speed = 850  # 8.50 km/h
        distance = 1234  # meters, uint24 LE
        data = struct.pack("<HH", flags, speed)
        # Append 3-byte distance
        data += bytes([distance & 0xFF, (distance >> 8) & 0xFF, (distance >> 16) & 0xFF])
        result = parse_treadmill_data(data)
        assert result.speed_kmh == 8.5
        assert result.distance_m == 1234

    def test_speed_and_incline(self) -> None:
        """Flags: speed + inclination (bit 3). Inclination + ramp angle = 4 bytes."""
        flags = 0x0008  # bit 3 set
        speed = 500  # 5.00 km/h
        incline = 35  # 3.5%
        ramp_angle = 20  # 2.0 degrees (skipped)
        data = struct.pack("<HHhh", flags, speed, incline, ramp_angle)
        result = parse_treadmill_data(data)
        assert result.speed_kmh == 5.0
        assert result.incline_pct == 3.5

    def test_negative_incline(self) -> None:
        """Negative inclination (decline)."""
        flags = 0x0008
        speed = 400
        incline = -20  # -2.0%
        ramp_angle = 0
        data = struct.pack("<HHhh", flags, speed, incline, ramp_angle)
        result = parse_treadmill_data(data)
        assert result.incline_pct == -2.0

    def test_speed_distance_incline_energy_hr_time(self) -> None:
        """Full payload with multiple fields."""
        # bits: 2 (distance), 3 (incline), 7 (energy), 8 (HR), 10 (elapsed)
        flags = 0x0004 | 0x0008 | 0x0080 | 0x0100 | 0x0400
        speed = 720  # 7.20 km/h
        data = struct.pack("<HH", flags, speed)
        # Distance: 5000m as uint24
        data += bytes([5000 & 0xFF, (5000 >> 8) & 0xFF, (5000 >> 16) & 0xFF])
        # Incline + ramp angle
        data += struct.pack("<hh", 50, 0)  # 5.0%, 0 ramp
        # Energy: total(uint16) + per_hour(uint16) + per_min(uint8) = 5 bytes
        data += struct.pack("<HHB", 250, 500, 8)
        # Heart rate: uint8
        data += bytes([145])
        # Elapsed time: uint16
        data += struct.pack("<H", 1800)

        result = parse_treadmill_data(data)
        assert result.speed_kmh == 7.2
        assert result.distance_m == 5000
        assert result.incline_pct == 5.0
        assert result.calories_kcal == 250
        assert result.heart_rate_bpm == 145
        assert result.elapsed_time_s == 1800

    def test_average_speed_skipped(self) -> None:
        """Bit 1 (average speed) is present but we skip it."""
        flags = 0x0002  # bit 1 set, bit 0 = 0 so speed present
        speed = 600
        avg_speed = 550
        data = struct.pack("<HHH", flags, speed, avg_speed)
        result = parse_treadmill_data(data)
        assert result.speed_kmh == 6.0


class TestParseFeatures:
    def test_empty(self) -> None:
        assert parse_features(b"") == []
        assert parse_features(b"\x00\x00\x00") == []

    def test_speed_and_incline_targets(self) -> None:
        # FM features: bit 2 (total_distance), bit 3 (inclination)
        fm = 0x0000_000C
        # Target features: bit 0 (speed_target), bit 1 (incline_target)
        ts = 0x0000_0003
        data = struct.pack("<II", fm, ts)
        features = parse_features(data)
        assert "total_distance" in features
        assert "inclination" in features
        assert "speed_target" in features
        assert "incline_target" in features


class TestParseMachineStatus:
    def test_known_status(self) -> None:
        assert parse_machine_status(bytes([0x04])) == "started_or_resumed"
        assert parse_machine_status(bytes([0x02])) == "stopped_by_user"

    def test_unknown_status(self) -> None:
        assert parse_machine_status(bytes([0xAB])) == "unknown_0xab"

    def test_empty(self) -> None:
        assert parse_machine_status(b"") == "unknown"


class TestParseHeartRate:
    def test_uint8_hr(self) -> None:
        """Flags bit 0 = 0: HR is uint8."""
        data = bytes([0x00, 72])
        assert parse_heart_rate(data) == 72

    def test_uint16_hr(self) -> None:
        """Flags bit 0 = 1: HR is uint16 LE."""
        data = bytes([0x01]) + struct.pack("<H", 165)
        assert parse_heart_rate(data) == 165
