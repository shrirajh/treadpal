from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TreadPalConfig:
    # Network
    host: str = "127.0.0.1"
    port: int = 8080

    # BLE
    scan_interval_s: float = 10.0
    scan_timeout_s: float = 5.0
    target_device_name: str | None = None
    target_device_address: str | None = None
    reconnect_delay_s: float = 5.0

    # Database
    db_path: str = "treadpal.db"

    # BPM Sync
    bpm_min_speed_kmh: float = 4.0
    bpm_max_speed_kmh: float = 7.0
    bpm_update_interval_s: float = 2.0
    bpm_harmonics: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)

    # Incline tracking (optional)
    home_incline: float | None = None  # Set to enable energy-based incline (e.g. 3.0)
    incline_range: float = 4.0  # Max ± offset from home

    # Treadmill quirks
    speed_send_mph: bool = False  # Treadmill interprets speed commands as mph
    speed_recv_mph: bool = False  # Treadmill reports speed in mph
    speed_step_kmh: float = 0.01  # Treadmill speed resolution in km/h (0.16 for 0.1mph treadmills)
    # Speed command resolution in the treadmill's command units. Commands are
    # rounded to this so the treadmill doesn't truncate them (2.49 mph -> 2.4).
    # None = 0.1 when speed_send_mph (mph treadmills step in 0.1), else 0.01.
    speed_cmd_resolution: float | None = None
    # Some treadmills report their target instead of the live speed/incline.
    # None = detect from impossible jumps; then estimate motion at these rates.
    estimate_motion: bool | None = None
    motion_speed_rate_kmh_s: float = 1.0
    motion_incline_rate_pct_s: float = 0.6

    # Fake treadmill for trying the UI without hardware
    simulate: bool = False

    # Logging: "debug" logs every BLE packet, control write and response
    log_level: str = "info"

    # Audio
    audio_sample_rate: int = 44100
    audio_hop_size: int = 256
    audio_win_size: int = 512
    audio_device_index: int | None = None

    # Stem separation (vocals/other/bass/drums) for the visualiser, ~40% of a CPU core
    stems: bool = True
    stems_threads: int = 2
    # Visualiser scale: lanes show from the song's recent peak down to this many dB
    # below it; quieter stem sound (and separation bleed) shows as nothing
    stems_floor_db: float = 36.0

    @property
    def speed_command_step(self) -> float:
        """Speed command resolution, in command units (mph if speed_send_mph)."""
        if self.speed_cmd_resolution is not None:
            return self.speed_cmd_resolution
        return 0.1 if self.speed_send_mph else 0.01

    @property
    def speed_resolution_kmh(self) -> float:
        step = self.speed_command_step
        return step * 1.60934 if self.speed_send_mph else step

    @classmethod
    def load(cls) -> TreadPalConfig:
        """Load config from env vars (TREADPAL_*) -> treadpal.json -> defaults."""
        kwargs: dict[str, Any] = {}

        # Try JSON config file
        config_path = Path("treadpal.json")
        if config_path.exists():
            with open(config_path) as f:
                file_config: dict[str, Any] = json.load(f)
            kwargs.update(file_config)

        # Env vars override file config
        env_map: dict[str, Any] = {
            "TREADPAL_HOST": str,
            "TREADPAL_PORT": int,
            "TREADPAL_SCAN_INTERVAL_S": float,
            "TREADPAL_SCAN_TIMEOUT_S": float,
            "TREADPAL_TARGET_DEVICE_NAME": str,
            "TREADPAL_TARGET_DEVICE_ADDRESS": str,
            "TREADPAL_RECONNECT_DELAY_S": float,
            "TREADPAL_DB_PATH": str,
            "TREADPAL_BPM_MIN_SPEED_KMH": float,
            "TREADPAL_BPM_MAX_SPEED_KMH": float,
            "TREADPAL_BPM_UPDATE_INTERVAL_S": float,
            "TREADPAL_SPEED_SEND_MPH": lambda v: v.lower() in ("true", "1", "yes"),
            "TREADPAL_SPEED_RECV_MPH": lambda v: v.lower() in ("true", "1", "yes"),
            "TREADPAL_AUDIO_SAMPLE_RATE": int,
            "TREADPAL_AUDIO_HOP_SIZE": int,
            "TREADPAL_AUDIO_WIN_SIZE": int,
            "TREADPAL_AUDIO_DEVICE_INDEX": int,
            "TREADPAL_SIMULATE": lambda v: v.lower() in ("true", "1", "yes"),
            "TREADPAL_LOG_LEVEL": str,
            "TREADPAL_STEMS": lambda v: v.lower() in ("true", "1", "yes"),
            "TREADPAL_STEMS_THREADS": int,
            "TREADPAL_STEMS_FLOOR_DB": float,
        }
        for env_key, cast in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                field_name = env_key.removeprefix("TREADPAL_").lower()
                kwargs[field_name] = cast(val)

        return cls(**kwargs)
