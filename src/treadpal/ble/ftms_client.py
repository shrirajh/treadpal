from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from bleak import BleakClient
from bleak.backends.characteristic import BleakGATTCharacteristic
from bleak.backends.device import BLEDevice

from treadpal.ble.ftms_protocol import (
    CONTROL_POINT_UUID,
    FEATURE_UUID,
    HR_MEASUREMENT_UUID,
    STATUS_UUID,
    TREADMILL_DATA_UUID,
    parse_features,
    parse_heart_rate,
    parse_machine_status,
    parse_treadmill_data,
)
from treadpal.db.queries import log_treadmill_data

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

            # Read supported features
            await self._read_features()

            # Subscribe to treadmill data
            await client.start_notify(
                TREADMILL_DATA_UUID, self._on_treadmill_data
            )

            # Subscribe to machine status
            await client.start_notify(STATUS_UUID, self._on_status_change)

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

    async def disconnect(self) -> None:
        if self._client is not None and self._client.is_connected:
            await self._client.disconnect()

    async def _read_features(self) -> None:
        assert self._client is not None
        raw = await self._client.read_gatt_char(FEATURE_UUID)
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
        parsed = parse_treadmill_data(data)
        if self._state.config.speed_is_mph:
            parsed = parsed.model_copy(
                update={"speed_kmh": round(parsed.speed_kmh * 1.60934, 2)}
            )
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
        logger.info("Machine status: %s", status)

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
        await self._client.write_gatt_char(CONTROL_POINT_UUID, data, response=True)

    async def _request_control(self) -> None:
        """Send Request Control opcode (0x00) before control commands."""
        await self._write_control(bytes([0x00]))

    async def start(self) -> None:
        await self._request_control()
        await self._write_control(bytes([0x07]))

    async def stop(self) -> None:
        await self._write_control(bytes([0x08, 0x01]))

    async def pause(self) -> None:
        await self._write_control(bytes([0x08, 0x02]))

    async def set_target_speed(self, speed_kmh: float) -> None:
        """Set target speed. FTMS uses 0.01 km/h resolution."""
        await self._request_control()
        speed = speed_kmh
        if self._state.config.speed_is_mph:
            speed = speed_kmh / 1.60934  # Convert km/h to mph for quirky firmware
        value = round(speed * 100)
        payload = bytes([0x02]) + value.to_bytes(2, "little", signed=False)
        await self._write_control(payload)

    async def set_target_incline(self, incline_pct: float) -> None:
        """Set target incline. FTMS uses 0.1% resolution, signed."""
        await self._request_control()
        value = int(incline_pct * 10)
        payload = bytes([0x03]) + value.to_bytes(2, "little", signed=True)
        await self._write_control(payload)
