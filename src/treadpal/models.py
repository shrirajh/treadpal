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


class TreadmillStatus(BaseModel):
    """Current connection and operational status."""

    connected: bool
    device_name: str | None
    device_address: str | None
    last_data: TreadmillData | None
    supported_features: list[str]


class ControlCommand(BaseModel):
    """Control command payload for set_speed / set_incline."""

    value: float


class BpmSyncConfig(BaseModel):
    """User-adjustable BPM sync parameters."""

    min_speed_kmh: float = 4.0
    max_speed_kmh: float = 7.0


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
