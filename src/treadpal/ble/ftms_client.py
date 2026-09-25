from __future__ import annotations

import asyncio
import logging
import struct
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from bleak import BleakClient
from bleak.backends.characteristic import BleakGATTCharacteristic
from bleak.backends.device import BLEDevice

from treadpal.ble.ftms_protocol import (
    CONTROL_POINT_UUID,
    FEATURE_UUID,
    HR_MEASUREMENT_UUID,
    INCLINE_RANGE_UUID,
    SPEED_RANGE_UUID,
    STATUS_UUID,
    TREADMILL_DATA_UUID,
    control_opcode_name,
    parse_control_response,
    parse_features,
    parse_heart_rate,
    parse_incline_range,
    parse_machine_status,
    parse_speed_range,
    parse_status_value,
    parse_treadmill_packet,
)
from treadpal.ble.motion import MotionEstimator, is_implausible_jump
from treadpal.db.queries import log_treadmill_data
from treadpal.models import TreadmillData, ValueRange

MPH_TO_KMH = 1.60934

if TYPE_CHECKING:
    from treadpal.app import AppState

logger = logging.getLogger("treadpal.ble")


class FTMSClient:
    """Manages FTMS BLE connection, notifications, and control commands."""

    def __init__(self, device: BLEDevice, state: AppState) -> None:
        self._device = device
        self._state = state
        self._client: BleakClient | None = None
        self._disconnect_event = asyncio.Event()
        self._warned_length = False
        self._init_motion()

    def _init_motion(self) -> None:
        cfg = self._state.config
        self._speed_est = MotionEstimator(cfg.motion_speed_rate_kmh_s)
        self._incline_est = MotionEstimator(cfg.motion_incline_rate_pct_s)
        self._last_reported: tuple[float, float, float] | None = None
        self._state.motion_estimated = bool(cfg.estimate_motion)

    @property
    def estimating(self) -> bool:
        return self._state.motion_estimated

    def current_data(self) -> TreadmillData | None:
        """Latest data, with estimated speed/incline advanced to now if estimating."""
        data = self._state.last_data
        if data is None or not self.estimating:
            return data
        now = time.monotonic()
        speed = self._speed_est.advance(now)
        incline = self._incline_est.advance(now)
        return data.model_copy(update={
            "speed_kmh": round(speed, 3) if speed is not None else data.speed_kmh,
            "incline_pct": round(incline, 2) if incline is not None else data.incline_pct,
        })

    def _track_reported_targets(self, fields: dict[str, Any]) -> None:
        """Treat reported speed/incline as targets; replace them with estimates."""
        now = time.monotonic()
        st = self._state
        speed = fields.get("speed_kmh")
        if speed is not None:
            if speed == 0 and st.machine_state in ("paused", "stopped"):
                # Belt stopped; keep the target so the UI can say what it resumes at
                self._speed_est.set_target(0.0, now)
            else:
                # Our own command is more precise than the (truncated) echo; take the
                # echo only when it disagrees, i.e. the console or a stop changed it
                if st.target_speed_kmh is None or abs(speed - st.target_speed_kmh) > 0.12:
                    st.target_speed_kmh = speed
                self._speed_est.set_target(st.target_speed_kmh, now)
            fields["speed_kmh"] = round(self._speed_est.value or 0.0, 3)
        incline = fields.get("incline_pct")
        if incline is not None:
            if st.target_incline_pct is None or abs(incline - st.target_incline_pct) > 0.05:
                st.target_incline_pct = incline
            self._incline_est.set_target(st.target_incline_pct, now)
            fields["incline_pct"] = round(self._incline_est.value or 0.0, 2)

    def _event(self, kind: str, **info: Any) -> None:
        """Record a BLE event for /api/debug/ble."""
        self._state.ble_events.append({"t": round(time.time(), 3), "kind": kind, **info})

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_connected

    @property
    def device_name(self) -> str | None:
        return self._device.name

    @property
    def device_address(self) -> str:
        return self._device.address

    def _on_disconnect(self, client: BleakClient) -> None:
        logger.info("Disconnected from %s", self._device.name)
        self._disconnect_event.set()

    async def connect_and_run(self) -> None:
        """Connect, read features, subscribe to notifications, block until disconnect."""
        logger.info("Connecting to %s [%s]", self._device.name, self._device.address)
        self._disconnect_event.clear()

        async with BleakClient(  # ty: ignore[invalid-context-manager]
            self._device, disconnected_callback=self._on_disconnect
        ) as client:
            self._client = client
            self._state.ftms_client = self

            logger.info("Connected to %s", self._device.name)
            self._event("connected", name=self._device.name, address=self._device.address)
            self._warned_length = False
            self._init_motion()
            if logger.isEnabledFor(logging.DEBUG):
                for service in client.services:
                    for char in service.characteristics:
                        logger.debug(
                            "GATT %s / %s %s", service.uuid, char.uuid, ",".join(char.properties)
                        )

            # Read supported features
            await self._read_features()
            await self._read_ranges()

            # Subscribe to treadmill data
            await client.start_notify(
                TREADMILL_DATA_UUID, self._on_treadmill_data
            )

            # Subscribe to machine status
            await client.start_notify(STATUS_UUID, self._on_status_change)

            # Control Point responses (indications) tell us whether commands were accepted
            try:
                await client.start_notify(CONTROL_POINT_UUID, self._on_control_response)
            except Exception:
                logger.info("Control point indications unavailable; command results won't be logged")

            # Optionally subscribe to HR if the service is available
            try:
                await client.start_notify(
                    HR_MEASUREMENT_UUID, self._on_heart_rate
                )
            except Exception:
                logger.debug("Heart rate service not available")

            # Block until disconnection
            await self._disconnect_event.wait()

        self._client = None
        self._state.ftms_client = None
        self._state.target_speed_kmh = None
        self._state.target_incline_pct = None
        self._state.machine_state = None

    async def disconnect(self) -> None:
        if self._client is not None and self._client.is_connected:
            await self._client.disconnect()

    async def _read_features(self) -> None:
        assert self._client is not None
        raw = await self._client.read_gatt_char(FEATURE_UUID)
        logger.debug("Feature raw: %s", raw.hex())
        self._event("features", hex=raw.hex())
        features = parse_features(raw)
        if features:
            self._state.supported_features = features
            logger.info("Supported features: %s", features)
        else:
            # Some treadmills return all-zeros on reconnect — infer features
            # from which GATT characteristics are actually present
            self._state.supported_features = self._infer_features()
            logger.info(
                "Feature read returned zeros, inferred from services: %s",
                self._state.supported_features,
            )

    async def _read_ranges(self) -> None:
        """Read supported speed/incline ranges; optional characteristics."""
        assert self._client is not None
        try:
            raw = await self._client.read_gatt_char(SPEED_RANGE_UUID)
            logger.debug("Speed range raw: %s -> %s", raw.hex(), parse_speed_range(raw))
            self._event("speed_range", hex=raw.hex())
            speed = parse_speed_range(raw)
            if speed is not None and speed[0] < speed[1]:
                # Range is in the treadmill's command units, same as target speed
                k = MPH_TO_KMH if self._state.config.speed_send_mph else 1.0
                self._state.speed_range = ValueRange(
                    min=round(speed[0] * k, 2), max=round(speed[1] * k, 2), step=round(speed[2] * k, 3)
                )
        except Exception:
            logger.debug("Speed range not available")
        try:
            raw = await self._client.read_gatt_char(INCLINE_RANGE_UUID)
            logger.debug("Incline range raw: %s -> %s", raw.hex(), parse_incline_range(raw))
            self._event("incline_range", hex=raw.hex())
            incline = parse_incline_range(raw)
            if incline is not None and incline[0] < incline[1]:
                self._state.incline_range = ValueRange(
                    min=incline[0], max=incline[1], step=incline[2]
                )
        except Exception:
            logger.debug("Incline range not available")
        logger.info(
            "Ranges: speed=%s incline=%s", self._state.speed_range, self._state.incline_range
        )

    def _infer_features(self) -> list[str]:
        """Infer supported features from available GATT characteristics."""
        assert self._client is not None
        features: list[str] = []
        services = {
            str(c.uuid): c for s in self._client.services for c in s.characteristics
        }
        if TREADMILL_DATA_UUID in services:
            features.extend([
                "average_speed", "total_distance", "inclination",
                "expended_energy", "elapsed_time",
            ])
        if CONTROL_POINT_UUID in services:
            features.extend(["speed_target", "incline_target"])
        if HR_MEASUREMENT_UUID in services:
            features.append("heart_rate")
        return features

    def _on_treadmill_data(
        self, _char: BleakGATTCharacteristic, data: bytearray
    ) -> None:
        try:
            pkt = parse_treadmill_packet(data)
        except (IndexError, struct.error) as e:
            logger.warning("Unparseable treadmill data %s: %s", data.hex(), e)
            self._event("bad_packet", hex=data.hex(), error=str(e))
            return

        fields = dict(pkt.fields)
        if "speed_kmh" in fields and self._state.config.speed_recv_mph:
            fields["speed_kmh"] = round(fields["speed_kmh"] * MPH_TO_KMH, 2)

        logger.debug(
            "Treadmill data %s flags=0x%04x %s fields=%s extras=%s",
            data.hex(), pkt.flags, pkt.flag_names, fields, pkt.extras,
        )
        self._state.ble_packets.append({
            "t": round(time.time(), 3),
            "hex": data.hex(),
            "flags": pkt.flag_names,
            "fields": fields,
            "extras": pkt.extras,
        })
        if pkt.consumed != len(data) and not self._warned_length:
            self._warned_length = True
            logger.warning(
                "Treadmill data is %d bytes but its flags describe %d; fields may be misread "
                "(raw %s, flags %s)",
                len(data), pkt.consumed, data.hex(), pkt.flag_names,
            )

        # Detect a treadmill that reports targets rather than live values
        cfg = self._state.config
        if "speed_kmh" in fields or "incline_pct" in fields:
            prev = self._last_reported
            cur = (
                time.monotonic(),
                fields.get("speed_kmh", prev[1] if prev else 0.0),
                fields.get("incline_pct", prev[2] if prev else 0.0),
            )
            if cfg.estimate_motion is None and not self.estimating:
                jump = is_implausible_jump(prev, cur)
                if jump:
                    self._state.motion_estimated = True
                    logger.info(
                        "Treadmill reports targets, not live values (%s); estimating motion "
                        "at %.2f km/h/s and %.2f %%/s (motion_*_rate settings)",
                        jump, cfg.motion_speed_rate_kmh_s, cfg.motion_incline_rate_pct_s,
                    )
            self._last_reported = cur
            if not self.estimating:
                # Live values: keep the estimators on them, so if a jump later
                # reveals target-reporting, estimation starts from the real position
                self._speed_est.sync(cur[1], cur[0])
                self._incline_est.sync(cur[2], cur[0])
        if self.estimating:
            self._track_reported_targets(fields)

        # Merge: a packet only updates the fields it carries
        base = self._state.last_data or TreadmillData(
            timestamp=datetime.now(timezone.utc), speed_kmh=0.0, incline_pct=0.0,
            distance_m=0, elapsed_time_s=0, calories_kcal=0, heart_rate_bpm=None,
        )
        parsed = base.model_copy(update={**fields, "timestamp": datetime.now(timezone.utc)})
        self._state.last_data = parsed

        async def _log() -> None:
            try:
                await log_treadmill_data(self._state.db, parsed)
            except Exception:
                logger.warning("Failed to log treadmill data", exc_info=True)

        asyncio.create_task(_log())

    def _on_status_change(
        self, _char: BleakGATTCharacteristic, data: bytearray
    ) -> None:
        status = parse_machine_status(data)
        value = parse_status_value(data)
        logger.info("Machine status: %s value=%s (raw %s)", status, value, data.hex())
        self._event("status", status=status, value=value, hex=data.hex())
        if status == "target_speed_changed" and value is not None:
            # Echoed in the treadmill's command units
            if self._state.config.speed_send_mph:
                value = round(value * MPH_TO_KMH, 2)
            self._state.target_speed_kmh = value
        elif status == "target_incline_changed" and value is not None:
            self._state.target_incline_pct = value
        elif status in ("stopped_by_user", "stopped_by_safety_key"):
            paused = len(data) > 1 and data[1] == 0x02
            self._state.machine_state = "paused" if paused else "stopped"
        elif status == "started_or_resumed":
            self._state.machine_state = "running"

    def _on_control_response(
        self, _char: BleakGATTCharacteristic, data: bytearray
    ) -> None:
        resp = parse_control_response(data)
        if resp is None:
            logger.debug("Control point indication (unrecognised): %s", data.hex())
            return
        opcode, result = resp
        self._event("control_response", opcode=opcode, result=result, hex=data.hex())
        if result == "success":
            logger.debug("Control %s -> %s", opcode, result)
        else:
            logger.warning("Control %s rejected by treadmill: %s", opcode, result)

    def _on_heart_rate(
        self, _char: BleakGATTCharacteristic, data: bytearray
    ) -> None:
        hr = parse_heart_rate(data)
        if self._state.last_data is not None:
            self._state.last_data = self._state.last_data.model_copy(
                update={"heart_rate_bpm": hr}
            )

    # --- Control commands ---

    async def _write_control(self, data: bytes) -> None:
        assert self._client is not None
        logger.debug("Control write %s: %s", control_opcode_name(data[0]), data.hex())
        self._event("control_write", opcode=control_opcode_name(data[0]), hex=data.hex())
        await self._client.write_gatt_char(CONTROL_POINT_UUID, data, response=True)

    async def _request_control(self) -> None:
        """Send Request Control opcode (0x00) before control commands."""
        await self._write_control(bytes([0x00]))

    async def start(self) -> None:
        await self._request_control()
        await self._write_control(bytes([0x07]))
        self._state.machine_state = "running"

    async def stop(self) -> None:
        await self._write_control(bytes([0x08, 0x01]))
        self._state.machine_state = "stopped"
        self._state.target_speed_kmh = None

    async def pause(self) -> None:
        await self._write_control(bytes([0x08, 0x02]))
        self._state.machine_state = "paused"

    async def set_target_speed(self, speed_kmh: float) -> None:
        """Set target speed. FTMS uses 0.01 km/h resolution (or 0.01 mph with speed_send_mph)."""
        cfg = self._state.config
        await self._request_control()
        speed = speed_kmh / MPH_TO_KMH if cfg.speed_send_mph else speed_kmh
        # Round to the treadmill's own step; left alone, it truncates (2.49 mph -> 2.4)
        step = cfg.speed_command_step
        speed = round(round(speed / step) * step, 2)
        value = round(speed * 100)
        payload = bytes([0x02]) + value.to_bytes(2, "little", signed=False)
        effective_kmh = round(speed * MPH_TO_KMH if cfg.speed_send_mph else speed, 3)
        logger.debug(
            "set_target_speed %.3f km/h -> %.2f %s (raw %d) = %.3f km/h effective",
            speed_kmh, speed, "mph" if cfg.speed_send_mph else "km/h", value, effective_kmh,
        )
        await self._write_control(payload)
        self._state.target_speed_kmh = effective_kmh
        self._speed_est.set_target(effective_kmh, time.monotonic())

    async def set_target_incline(self, incline_pct: float) -> None:
        """Set target incline. FTMS uses 0.1% resolution, signed."""
        await self._request_control()
        value = round(incline_pct * 10)
        payload = bytes([0x03]) + value.to_bytes(2, "little", signed=True)
        logger.debug("set_target_incline %.1f%% (raw %d)", incline_pct, value)
        await self._write_control(payload)
        self._state.target_incline_pct = value / 10
        self._incline_est.set_target(value / 10, time.monotonic())
