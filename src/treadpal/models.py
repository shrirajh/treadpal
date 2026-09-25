from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TreadmillData(BaseModel):
    """Parsed data from FTMS Treadmill Data characteristic (0x2ACD)."""

    timestamp: datetime
    speed_kmh: float
    incline_pct: float
    distance_m: int
    elapsed_time_s: int
    calories_kcal: int
    heart_rate_bpm: int | None


class ValueRange(BaseModel):
    """Supported min/max/step for a controllable value."""

    min: float
    max: float
    step: float


class TreadmillStatus(BaseModel):
    """Current connection and operational status."""

    connected: bool
    device_name: str | None
    device_address: str | None
    last_data: TreadmillData | None
    supported_features: list[str]
    # Last known targets: from our own commands or the treadmill's status notifications
    target_speed_kmh: float | None = None
    target_incline_pct: float | None = None
    machine_state: str | None = None  # "running" | "paused" | "stopped"
    speed_range: ValueRange | None = None
    incline_range: ValueRange | None = None
    prefers_mph: bool = False
    speed_resolution_kmh: float = 0.01  # Smallest speed change the treadmill makes
    motion_estimated: bool = False  # last_data speed/incline are estimates (treadmill reports targets)


class ControlCommand(BaseModel):
    """Control command payload for set_speed / set_incline."""

    value: float


class BpmSyncConfig(BaseModel):
    """User-adjustable BPM sync parameters."""

    min_speed_kmh: float = 4.0
    max_speed_kmh: float = 7.0
    harmonics: list[float] | None = None


class BpmUpdate(BaseModel):
    """External BPM source payload."""

    bpm: float = Field(ge=30, le=250)
    incline_pct: float | None = Field(default=None, ge=-10, le=40)


class BpmSyncStatus(BaseModel):
    """Current BPM sync state."""

    active: bool
    detected_bpm: float | None
    selected_harmonic: float | None
    effective_cadence: float | None
    implied_stride_m: float | None
    natural_stride_m: float | None
    stride_score: float | None
    commanded_speed_kmh: float | None
    min_speed_kmh: float
    max_speed_kmh: float
    paused: bool = False
    audio_clients: int = 0
    stems: str = "off"  # Stem separation: off, preparing, ready, error
    harmonic_override: bool = False
    harmonics: list[float] = []
    age_s: float | None = None  # Seconds since the last BPM result


class HistorySummary(BaseModel):
    """Aggregated session summary."""

    session_start: datetime
    session_end: datetime
    duration_s: int
    distance_m: int
    avg_speed_kmh: float
    max_speed_kmh: float
    avg_incline_pct: float
    total_calories: int
    avg_heart_rate: int | None
