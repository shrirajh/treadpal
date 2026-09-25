"""Fake FTMS treadmill for trying the UI without hardware (TREADPAL_SIMULATE=1).

Commands go through the real FTMSClient code paths; only the BLE writes are
dropped. It behaves like a target-reporting treadmill: it
reports the *target* speed/incline rather than live values, and distance in
10 m steps. Real Treadmill Data packets are built and fed to the normal
notification handler, so parsing and motion estimation run as they would.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from typing import TYPE_CHECKING

from treadpal.ble.ftms_client import MPH_TO_KMH, FTMSClient
from treadpal.models import ValueRange

if TYPE_CHECKING:
    from treadpal.app import AppState

logger = logging.getLogger("treadpal.sim")

TICK_S = 0.9  # The real one notifies a little faster than 1 Hz
ACCEL_KMH_S = 0.8  # "True" motor rates, used only to integrate distance
INCLINE_PCT_S = 0.5
START_SPEED_KMH = 1.29  # 0.8 mph
DISTANCE_STEP_M = 10


def build_treadmill_packet(
    speed_kmh: float, distance_m: int, incline_pct: float, kcal: int, elapsed_s: int
) -> bytes:
    # speed (implicit), distance, inclination + ramp angle, energy, elapsed time
    flags = (1 << 2) | (1 << 3) | (1 << 7) | (1 << 10)
    return (
        struct.pack("<HH", flags, round(speed_kmh * 100))
        + distance_m.to_bytes(3, "little")
        + struct.pack("<hh", round(incline_pct * 10), 5)
        + struct.pack("<HHB", kcal, 0, 0)
        + struct.pack("<H", elapsed_s)
    )


class SimulatedFTMSClient(FTMSClient):
    def __init__(self, state: AppState) -> None:
        self._state = state
        self._client = None
        self._disconnect_event = asyncio.Event()
        self._warned_length = False
        self._init_motion()
        # The treadmill's own memory, driven only by control point writes
        self._running = False
        self._paused = False
        self._target_speed = 0.0
        self._target_incline = 0.0

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def device_name(self) -> str | None:
        return "Simulated Treadmill"

    @property
    def device_address(self) -> str:
        return "SIM"

    async def _write_control(self, data: bytes) -> None:
        """Act on the bytes like the hardware would."""
        logger.debug("sim control write: %s", data.hex())
        op = data[0]
        if op == 0x02:
            v = int.from_bytes(data[1:3], "little") / 100
            self._target_speed = v * MPH_TO_KMH if self._state.config.speed_send_mph else v
        elif op == 0x03:
            self._target_incline = int.from_bytes(data[1:3], "little", signed=True) / 10
        elif op == 0x07:
            if not self._running and not self._paused:
                self._reset_session()
            self._running, self._paused = True, False
            if self._target_speed <= 0:
                self._target_speed = START_SPEED_KMH
        elif op == 0x08:
            self._running = False
            self._paused = data[1] == 0x02
            if not self._paused:
                self._target_speed = 0.0

    async def disconnect(self) -> None:
        pass

    def _reset_session(self) -> None:
        self._distance_m = 0.0
        self._elapsed_s = 0.0
        self._kcal = 0.0

    async def run(self) -> None:
        state = self._state
        state.ftms_client = self
        state.machine_state = "stopped"
        state.supported_features = [
            "total_distance", "inclination", "expended_energy",
            "elapsed_time", "speed_target", "incline_target",
        ]
        state.speed_range = ValueRange(min=START_SPEED_KMH, max=12.0, step=0.1)
        state.incline_range = ValueRange(min=0.0, max=12.0, step=0.5)
        speed = 0.0
        incline = 0.0
        self._reset_session()

        while True:
            running = self._running
            target_speed = self._target_speed if running else 0.0
            target_incline = self._target_incline

            speed = _approach(speed, target_speed, ACCEL_KMH_S * TICK_S)
            incline = _approach(incline, target_incline, INCLINE_PCT_S * TICK_S)
            if running:
                self._elapsed_s += TICK_S
            self._distance_m += speed / 3.6 * TICK_S
            # Rough walking energy: ~1 kcal/kg/km, +grade cost, 75 kg walker
            self._kcal += speed / 3600 * TICK_S * 75 * (1 + incline / 10)

            # Report like the real thing: targets, truncated speed, coarse distance
            packet = build_treadmill_packet(
                int(target_speed * 10) / 10,
                int(self._distance_m // DISTANCE_STEP_M * DISTANCE_STEP_M),
                target_incline,
                int(self._kcal),
                int(self._elapsed_s),
            )
            self._on_treadmill_data(None, bytearray(packet))  # type: ignore[arg-type]
            await asyncio.sleep(TICK_S)


def _approach(value: float, target: float, max_step: float) -> float:
    diff = target - value
    if abs(diff) <= max_step:
        return target
    return value + (max_step if diff > 0 else -max_step)


async def run_simulator(state: AppState) -> None:
    logger.info("Running with simulated treadmill")
    await SimulatedFTMSClient(state).run()
