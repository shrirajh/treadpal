"""FTMS (Fitness Machine Service) binary protocol parsing.

Handles Treadmill Data (0x2ACD), Fitness Machine Feature (0x2ACC),
Fitness Machine Status (0x2ADA), and Heart Rate Measurement (0x2A37).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from treadpal.models import TreadmillData

# FTMS UUIDs
FTMS_SERVICE_UUID = "00001826-0000-1000-8000-00805f9b34fb"
TREADMILL_DATA_UUID = "00002acd-0000-1000-8000-00805f9b34fb"
CONTROL_POINT_UUID = "00002ad9-0000-1000-8000-00805f9b34fb"
FEATURE_UUID = "00002acc-0000-1000-8000-00805f9b34fb"
STATUS_UUID = "00002ada-0000-1000-8000-00805f9b34fb"
HR_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
SPEED_RANGE_UUID = "00002ad4-0000-1000-8000-00805f9b34fb"
INCLINE_RANGE_UUID = "00002ad5-0000-1000-8000-00805f9b34fb"


TREADMILL_DATA_FLAG_NAMES = [
    "more_data",  # Bit 0 is inverted: set means speed is NOT in this packet
    "average_speed",
    "total_distance",
    "inclination_and_ramp_angle",
    "elevation_gain",
    "instantaneous_pace",
    "average_pace",
    "expended_energy",
    "heart_rate",
    "metabolic_equivalent",
    "elapsed_time",
    "remaining_time",
    "force_and_power",
]


@dataclass
class TreadmillPacket:
    """One Treadmill Data notification, keeping track of which fields it carried.

    Treadmills may split data across notifications, so absent fields must not
    be read as zero. ``fields`` uses TreadmillData names; ``extras`` holds
    parsed values we don't model (useful for debugging).
    """

    flags: int
    fields: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    consumed: int = 2  # Bytes the flags say the packet should have

    @property
    def flag_names(self) -> list[str]:
        return [n for i, n in enumerate(TREADMILL_DATA_FLAG_NAMES) if self.flags & (1 << i)]


def parse_treadmill_packet(data: bytes | bytearray) -> TreadmillPacket:
    """Parse FTMS Treadmill Data characteristic (0x2ACD).

    The first 2 bytes are flags (little-endian). Bit 0 is INVERTED:
    0 means Instantaneous Speed IS present. Fields are variable-length
    and must be consumed in order based on which flag bits are set.
    Raises struct.error / IndexError if the packet is shorter than its flags claim.
    """
    flags = struct.unpack_from("<H", data, 0)[0]
    pkt = TreadmillPacket(flags=flags)
    f, x = pkt.fields, pkt.extras
    offset = 2

    if not (flags & (1 << 0)):
        f["speed_kmh"] = struct.unpack_from("<H", data, offset)[0] / 100.0
        offset += 2

    if flags & (1 << 1):
        x["average_speed_kmh"] = struct.unpack_from("<H", data, offset)[0] / 100.0
        offset += 2

    # Total Distance: uint24
    if flags & (1 << 2):
        f["distance_m"] = data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16)
        offset += 3

    # Inclination (sint16, 0.1%) + Ramp Angle Setting (sint16, 0.1 degree)
    if flags & (1 << 3):
        f["incline_pct"] = struct.unpack_from("<h", data, offset)[0] / 10.0
        x["ramp_angle_deg"] = struct.unpack_from("<h", data, offset + 2)[0] / 10.0
        offset += 4

    # Elevation Gain: positive uint16 + negative uint16, 0.1 m
    if flags & (1 << 4):
        pos, neg = struct.unpack_from("<HH", data, offset)
        x["elevation_gain_m"] = (pos / 10.0, neg / 10.0)
        offset += 4

    # Instantaneous / Average Pace: uint16
    if flags & (1 << 5):
        x["instantaneous_pace"] = struct.unpack_from("<H", data, offset)[0]
        offset += 2
    if flags & (1 << 6):
        x["average_pace"] = struct.unpack_from("<H", data, offset)[0]
        offset += 2

    # Expended Energy: total uint16 + per hour uint16 + per minute uint8
    if flags & (1 << 7):
        f["calories_kcal"] = struct.unpack_from("<H", data, offset)[0]
        offset += 5

    if flags & (1 << 8):
        # 0 means no strap is connected
        f["heart_rate_bpm"] = data[offset] or None
        offset += 1

    if flags & (1 << 9):
        x["metabolic_equivalent"] = data[offset] / 10.0
        offset += 1

    if flags & (1 << 10):
        f["elapsed_time_s"] = struct.unpack_from("<H", data, offset)[0]
        offset += 2

    if flags & (1 << 11):
        x["remaining_time_s"] = struct.unpack_from("<H", data, offset)[0]
        offset += 2

    # Force on Belt (sint16, N) + Power Output (sint16, W)
    if flags & (1 << 12):
        x["force_n"], x["power_w"] = struct.unpack_from("<hh", data, offset)
        offset += 4

    pkt.consumed = offset
    return pkt


def parse_treadmill_data(data: bytes | bytearray) -> TreadmillData:
    """Parse a Treadmill Data packet, defaulting absent fields to zero."""
    fields = parse_treadmill_packet(data).fields
    return TreadmillData(
        timestamp=datetime.now(timezone.utc),
        speed_kmh=fields.get("speed_kmh", 0.0),
        incline_pct=fields.get("incline_pct", 0.0),
        distance_m=fields.get("distance_m", 0),
        elapsed_time_s=fields.get("elapsed_time_s", 0),
        calories_kcal=fields.get("calories_kcal", 0),
        heart_rate_bpm=fields.get("heart_rate_bpm"),
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


def parse_status_value(data: bytes | bytearray) -> float | None:
    """Parse the parameter of a target-changed Machine Status event.

    0x05 target speed: uint16, 0.01 km/h. 0x06 target incline: sint16, 0.1%.
    Returns None for other opcodes or truncated payloads.
    """
    if len(data) < 3:
        return None
    if data[0] == 0x05:
        return struct.unpack_from("<H", data, 1)[0] / 100.0
    if data[0] == 0x06:
        return struct.unpack_from("<h", data, 1)[0] / 10.0
    return None


def parse_speed_range(data: bytes | bytearray) -> tuple[float, float, float] | None:
    """Parse Supported Speed Range (0x2AD4): min, max, step as uint16 in 0.01 km/h."""
    if len(data) < 6:
        return None
    lo, hi, step = struct.unpack_from("<HHH", data, 0)
    return lo / 100.0, hi / 100.0, step / 100.0


def parse_incline_range(data: bytes | bytearray) -> tuple[float, float, float] | None:
    """Parse Supported Inclination Range (0x2AD5): sint16 min, sint16 max, uint16 step in 0.1%."""
    if len(data) < 6:
        return None
    lo, hi, step = struct.unpack_from("<hhH", data, 0)
    return lo / 10.0, hi / 10.0, step / 10.0


_CONTROL_OPCODES: dict[int, str] = {
    0x00: "request_control",
    0x01: "reset",
    0x02: "set_target_speed",
    0x03: "set_target_inclination",
    0x07: "start_or_resume",
    0x08: "stop_or_pause",
}

_CONTROL_RESULTS: dict[int, str] = {
    0x01: "success",
    0x02: "op_code_not_supported",
    0x03: "invalid_parameter",
    0x04: "operation_failed",
    0x05: "control_not_permitted",
}


def control_opcode_name(opcode: int) -> str:
    return _CONTROL_OPCODES.get(opcode, f"opcode_0x{opcode:02x}")


def parse_control_response(data: bytes | bytearray) -> tuple[str, str] | None:
    """Parse a Control Point indication: 0x80, request opcode, result code."""
    if len(data) < 3 or data[0] != 0x80:
        return None
    return (
        control_opcode_name(data[1]),
        _CONTROL_RESULTS.get(data[2], f"result_0x{data[2]:02x}"),
    )


def parse_heart_rate(data: bytes | bytearray) -> int:
    """Parse Heart Rate Measurement (0x2A37).

    Bit 0 of flags: 0 = HR is uint8 at offset 1, 1 = HR is uint16 LE at offset 1.
    """
    flags = data[0]
    if flags & 0x01:
        return struct.unpack_from("<H", data, 1)[0]
    return data[1]
