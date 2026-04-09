"""FTMS (Fitness Machine Service) binary protocol parsing.

Handles Treadmill Data (0x2ACD), Fitness Machine Feature (0x2ACC),
Fitness Machine Status (0x2ADA), and Heart Rate Measurement (0x2A37).
"""

from __future__ import annotations

import struct
from datetime import datetime, timezone

from treadpal.models import TreadmillData

# FTMS UUIDs
FTMS_SERVICE_UUID = "00001826-0000-1000-8000-00805f9b34fb"
TREADMILL_DATA_UUID = "00002acd-0000-1000-8000-00805f9b34fb"
CONTROL_POINT_UUID = "00002ad9-0000-1000-8000-00805f9b34fb"
FEATURE_UUID = "00002acc-0000-1000-8000-00805f9b34fb"
STATUS_UUID = "00002ada-0000-1000-8000-00805f9b34fb"
HR_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"


def parse_treadmill_data(data: bytes | bytearray) -> TreadmillData:
    """Parse FTMS Treadmill Data characteristic (0x2ACD).

    The first 2 bytes are flags (little-endian). Bit 0 is INVERTED:
    0 means Instantaneous Speed IS present. Fields are variable-length
    and must be consumed in order based on which flag bits are set.
    """
    flags = struct.unpack_from("<H", data, 0)[0]
    offset = 2

    # Bit 0 is INVERTED: 0 = speed present, 1 = speed NOT present
    speed_kmh = 0.0
    if not (flags & (1 << 0)):
        speed_kmh = struct.unpack_from("<H", data, offset)[0] / 100.0
        offset += 2

    # Bit 1: Average Speed (uint16, skip)
    if flags & (1 << 1):
        offset += 2

    # Bit 2: Total Distance (uint24 LE, 3 bytes)
    distance_m = 0
    if flags & (1 << 2):
        distance_m = data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16)
        offset += 3

    # Bit 3: Inclination (int16 LE) + Ramp Angle (int16 LE)
    incline_pct = 0.0
    if flags & (1 << 3):
        incline_pct = struct.unpack_from("<h", data, offset)[0] / 10.0
        offset += 2
        offset += 2  # skip ramp angle

    # Bit 4: Elevation Gain (positive uint16 + negative uint16)
    if flags & (1 << 4):
        offset += 4

    # Bit 5: Instantaneous Pace (uint16, skip)
    if flags & (1 << 5):
        offset += 2

    # Bit 6: Average Pace (uint16, skip)
    if flags & (1 << 6):
        offset += 2

    # Bit 7: Expended Energy (total uint16 + per_hour uint16 + per_min uint8 = 5 bytes)
    calories_kcal = 0
    if flags & (1 << 7):
        calories_kcal = struct.unpack_from("<H", data, offset)[0]
        offset += 5

    # Bit 8: Heart Rate (uint8)
    heart_rate_bpm: int | None = None
    if flags & (1 << 8):
        heart_rate_bpm = data[offset]
        offset += 1

    # Bit 9: Metabolic Equivalent (uint8, skip)
    if flags & (1 << 9):
        offset += 1

    # Bit 10: Elapsed Time (uint16 LE, seconds)
    elapsed_time_s = 0
    if flags & (1 << 10):
        elapsed_time_s = struct.unpack_from("<H", data, offset)[0]
        offset += 2

    # Bits 11-12: Remaining Time, Force/Power — skip

    return TreadmillData(
        timestamp=datetime.now(timezone.utc),
        speed_kmh=speed_kmh,
        incline_pct=incline_pct,
        distance_m=distance_m,
        elapsed_time_s=elapsed_time_s,
        calories_kcal=calories_kcal,
        heart_rate_bpm=heart_rate_bpm,
    )


_FM_FEATURE_NAMES = [
    "average_speed",
    "cadence",
    "total_distance",
    "inclination",
    "elevation_gain",
    "pace",
    "step_count",
    "resistance_level",
    "stair_count",
    "expended_energy",
    "heart_rate",
    "metabolic_equivalent",
    "elapsed_time",
    "remaining_time",
    "power_measurement",
    "force_on_belt",
    "user_data_retention",
]

_TS_FEATURE_NAMES = [
    "speed_target",
    "incline_target",
    "resistance_target",
    "power_target",
    "heart_rate_target",
    "targeted_expended_energy",
    "targeted_step_count",
    "targeted_stride_count",
    "targeted_distance",
    "targeted_training_time",
    "targeted_time_in_two_hr_zones",
    "targeted_time_in_three_hr_zones",
    "targeted_time_in_five_hr_zones",
    "indoor_bike_simulation",
    "wheel_circumference",
    "spin_down_control",
    "targeted_cadence",
]


def parse_features(data: bytes | bytearray) -> list[str]:
    """Parse FTMS Fitness Machine Feature (0x2ACC).

    First 4 bytes = Fitness Machine Features bitfield.
    Next 4 bytes = Target Setting Features bitfield.
    """
    if len(data) < 8:
        return []

    fm_features = struct.unpack_from("<I", data, 0)[0]
    ts_features = struct.unpack_from("<I", data, 4)[0]

    result: list[str] = []
    for i, name in enumerate(_FM_FEATURE_NAMES):
        if fm_features & (1 << i):
            result.append(name)
    for i, name in enumerate(_TS_FEATURE_NAMES):
        if ts_features & (1 << i):
            result.append(name)
    return result


_STATUS_MAP: dict[int, str] = {
    0x01: "reset",
    0x02: "stopped_by_user",
    0x03: "stopped_by_safety_key",
    0x04: "started_or_resumed",
    0x05: "target_speed_changed",
    0x06: "target_incline_changed",
    0x07: "target_resistance_changed",
    0x08: "target_power_changed",
    0x09: "target_heart_rate_changed",
    0x0A: "targeted_expended_energy_changed",
    0x0B: "targeted_step_count_changed",
    0x0C: "targeted_stride_count_changed",
    0x0D: "targeted_distance_changed",
    0x0E: "targeted_training_time_changed",
    0xFF: "control_permission_lost",
}


def parse_machine_status(data: bytes | bytearray) -> str:
    """Parse FTMS Fitness Machine Status (0x2ADA)."""
    if not data:
        return "unknown"
    return _STATUS_MAP.get(data[0], f"unknown_0x{data[0]:02x}")


def parse_heart_rate(data: bytes | bytearray) -> int:
    """Parse Heart Rate Measurement (0x2A37).

    Bit 0 of flags: 0 = HR is uint8 at offset 1, 1 = HR is uint16 LE at offset 1.
    """
    flags = data[0]
    if flags & 0x01:
        return struct.unpack_from("<H", data, 1)[0]
    return data[1]
