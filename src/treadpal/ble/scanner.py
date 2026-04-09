from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from bleak import BleakScanner
from bleak.backends.device import BLEDevice

from treadpal.ble.ftms_client import FTMSClient
from treadpal.ble.ftms_protocol import FTMS_SERVICE_UUID

if TYPE_CHECKING:
    from treadpal.app import AppState

logger = logging.getLogger("treadpal.ble")


async def scan_for_treadmill(
    config: object,
) -> BLEDevice | None:
    """Scan for a BLE device advertising FTMS service."""
    from treadpal.config import TreadPalConfig

    assert isinstance(config, TreadPalConfig)

    devices = await BleakScanner.discover(
        timeout=config.scan_timeout_s,
        service_uuids=[FTMS_SERVICE_UUID],
    )

    for d in devices:
        if config.target_device_name and config.target_device_name not in (
            d.name or ""
        ):
            continue
        if config.target_device_address and d.address != config.target_device_address:
            continue
        logger.info("Found FTMS device: %s [%s]", d.name, d.address)
        return d

    if devices:
        d = devices[0]
        logger.info("Found FTMS device: %s [%s]", d.name, d.address)
        return d

    return None


async def run_scanner_loop(state: AppState) -> None:
    """Continuously scan for FTMS treadmills, connect when found."""
    logger.info("BLE scanner started")

    while True:
        # Skip scanning if already connected
        if state.ftms_client is not None:
            assert isinstance(state.ftms_client, FTMSClient)
            if state.ftms_client.is_connected:
                await asyncio.sleep(state.config.scan_interval_s)
                continue

        logger.debug("Scanning for FTMS devices...")
        device = await scan_for_treadmill(state.config)

        if device is not None:
            client = FTMSClient(device, state)
            try:
                await client.connect_and_run()
            except Exception:
                logger.exception("FTMS connection lost")
                state.ftms_client = None

        await asyncio.sleep(state.config.reconnect_delay_s)
