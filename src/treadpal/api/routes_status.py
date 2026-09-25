from __future__ import annotations

from fastapi import APIRouter, Request

from treadpal.app import get_state
from treadpal.models import TreadmillStatus

router = APIRouter(tags=["status"])


@router.get("/status")
async def get_status(request: Request) -> TreadmillStatus:
    state = get_state(request.app)
    connected = False
    device_name: str | None = None
    device_address: str | None = None
    data = state.last_data

    if state.ftms_client is not None:
        from treadpal.ble.ftms_client import FTMSClient

        assert isinstance(state.ftms_client, FTMSClient)
        connected = bool(state.ftms_client.is_connected)
        device_name = state.ftms_client.device_name
        device_address = state.ftms_client.device_address
        data = state.ftms_client.current_data()

    return TreadmillStatus(
        connected=connected,
        device_name=device_name,
        device_address=device_address,
        last_data=data,
        supported_features=state.supported_features,
        target_speed_kmh=state.target_speed_kmh,
        target_incline_pct=state.target_incline_pct,
        machine_state=state.machine_state,
        speed_range=state.speed_range,
        incline_range=state.incline_range,
        prefers_mph=state.config.speed_send_mph or state.config.speed_recv_mph,
        speed_resolution_kmh=state.config.speed_resolution_kmh,
        motion_estimated=state.motion_estimated,
    )


@router.get("/features")
async def get_features(request: Request) -> list[str]:
    state = get_state(request.app)
    return state.supported_features


@router.get("/debug/ble")
async def debug_ble(request: Request) -> dict[str, object]:
    """Recent raw BLE traffic: treadmill data packets (newest last) and events
    (status notifications, control writes and the treadmill's responses)."""
    state = get_state(request.app)
    return {
        "config": {
            "speed_send_mph": state.config.speed_send_mph,
            "speed_recv_mph": state.config.speed_recv_mph,
            "speed_command_step": state.config.speed_command_step,
        },
        "packets": list(state.ble_packets),
        "events": list(state.ble_events),
    }


@router.get("/devices")
async def scan_devices(request: Request) -> list[dict[str, str | None]]:
    """One-shot scan returning visible FTMS devices."""
    from bleak import BleakScanner

    from treadpal.ble.ftms_protocol import FTMS_SERVICE_UUID

    state = get_state(request.app)
    devices = await BleakScanner.discover(
        timeout=state.config.scan_timeout_s,
        service_uuids=[FTMS_SERVICE_UUID],
    )
    return [{"name": d.name, "address": d.address} for d in devices]
