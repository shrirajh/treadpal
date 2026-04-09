from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from treadpal.app import get_state
from treadpal.models import ControlCommand

router = APIRouter(tags=["control"])


def _get_ftms_client(request: Request):  # noqa: ANN202
    from treadpal.ble.ftms_client import FTMSClient

    state = get_state(request.app)
    if state.ftms_client is None or not isinstance(state.ftms_client, FTMSClient):
        raise HTTPException(status_code=409, detail="Treadmill not connected")
    if not state.ftms_client.is_connected:
        raise HTTPException(status_code=409, detail="Treadmill not connected")
    return state.ftms_client


@router.post("/control/start")
async def control_start(request: Request) -> dict[str, str]:
    client = _get_ftms_client(request)
    await client.start()
    return {"status": "ok"}


@router.post("/control/stop")
async def control_stop(request: Request) -> dict[str, str]:
    client = _get_ftms_client(request)
    await client.stop()
    return {"status": "ok"}


@router.post("/control/pause")
async def control_pause(request: Request) -> dict[str, str]:
    client = _get_ftms_client(request)
    await client.pause()
    return {"status": "ok"}


@router.post("/control/set_speed")
async def control_set_speed(request: Request, body: ControlCommand) -> dict[str, str]:
    if body.value < 0 or body.value > 25:
        raise HTTPException(status_code=400, detail="Speed must be 0-25 km/h")
    client = _get_ftms_client(request)
    await client.set_target_speed(body.value)
    return {"status": "ok"}


@router.post("/control/set_incline")
async def control_set_incline(
    request: Request, body: ControlCommand
) -> dict[str, str]:
    if body.value < -10 or body.value > 40:
        raise HTTPException(status_code=400, detail="Incline must be -10 to 40%")
    client = _get_ftms_client(request)
    await client.set_target_incline(body.value)
    return {"status": "ok"}
