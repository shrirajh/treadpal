from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

from treadpal.app import get_state
from treadpal.audio.bpm_sync import BpmSyncController
from treadpal.models import BpmSyncConfig, BpmSyncStatus, BpmUpdate

logger = logging.getLogger("treadpal.bpm")

router = APIRouter(tags=["bpm"])


def _make_controller(
    config: BpmSyncConfig | None, cfg: object
) -> BpmSyncController:
    from treadpal.config import TreadPalConfig

    assert isinstance(cfg, TreadPalConfig)
    return BpmSyncController(
        min_speed_kmh=config.min_speed_kmh if config else cfg.bpm_min_speed_kmh,
        max_speed_kmh=config.max_speed_kmh if config else cfg.bpm_max_speed_kmh,
        harmonics=cfg.bpm_harmonics,
        speed_step_kmh=cfg.speed_step_kmh,
    )


@router.post("/bpm/start")
async def start_bpm_sync(
    request: Request, config: BpmSyncConfig | None = None
) -> dict[str, str]:
    """Start local audio capture + BPM sync."""
    state = get_state(request.app)

    if state.bpm_task is not None and not state.bpm_task.done():
        raise HTTPException(status_code=409, detail="BPM sync already running")

    cfg = state.config
    state.bpm_sync = _make_controller(config, cfg)

    from treadpal.audio.capture import AudioCapture
    from treadpal.audio.bpm_detect import BpmDetector

    async def _run_bpm_sync() -> None:
        capture = AudioCapture(
            sample_rate=cfg.audio_sample_rate,
            hop_size=cfg.audio_hop_size,
            device_index=cfg.audio_device_index,
        )
        detector = BpmDetector(
            sample_rate=cfg.audio_sample_rate,
            win_size=cfg.audio_win_size,
            hop_size=cfg.audio_hop_size,
        )
        try:
            capture.start()
        except Exception:
            logger.exception("Failed to start audio capture")
            return
        last_command_time = 0.0
        try:
            while True:
                block = await capture.read_block()
                bpm = detector.process(block)
                if bpm is not None and bpm > 0 and state.bpm_sync is not None:
                    result = state.bpm_sync.compute(bpm)
                    now = asyncio.get_event_loop().time()
                    if (now - last_command_time) >= cfg.bpm_update_interval_s:
                        if state.ftms_client is not None:
                            from treadpal.ble.ftms_client import FTMSClient

                            assert isinstance(state.ftms_client, FTMSClient)
                            if state.ftms_client.is_connected:
                                await state.ftms_client.set_target_speed(
                                    result.speed_kmh
                                )
                                last_command_time = now
        finally:
            capture.stop()

    state.bpm_task = asyncio.create_task(_run_bpm_sync(), name="bpm-sync")
    return {"status": "started"}


@router.post("/bpm/stop")
async def stop_bpm_sync(request: Request) -> dict[str, str]:
    """Stop local BPM sync."""
    state = get_state(request.app)
    if state.bpm_task is None or state.bpm_task.done():
        raise HTTPException(status_code=409, detail="BPM sync not running")

    state.bpm_task.cancel()
    try:
        await state.bpm_task
    except asyncio.CancelledError:
        pass
    state.bpm_task = None
    return {"status": "stopped"}


@router.post("/bpm/update")
async def update_bpm(request: Request, body: BpmUpdate) -> dict[str, object]:
    """Accept BPM from an external source, run through harmonic snapper."""
    state = get_state(request.app)

    if state.bpm_sync is None:
        state.bpm_sync = _make_controller(None, state.config)

    result = state.bpm_sync.compute(body.bpm)

    # Send to treadmill if connected
    if state.ftms_client is not None:
        from treadpal.ble.ftms_client import FTMSClient

        assert isinstance(state.ftms_client, FTMSClient)
        if state.ftms_client.is_connected:
            await state.ftms_client.set_target_speed(result.speed_kmh)
            if body.incline_pct is not None:
                await state.ftms_client.set_target_incline(body.incline_pct)

    resp: dict[str, object] = {
        "detected_bpm": result.detected_bpm,
        "selected_harmonic": result.selected_harmonic,
        "effective_cadence": result.effective_cadence,
        "implied_stride_m": result.implied_stride_m,
        "natural_stride_m": result.natural_stride_m,
        "stride_score": result.stride_score,
        "commanded_speed_kmh": result.speed_kmh,
    }
    if body.incline_pct is not None:
        resp["commanded_incline_pct"] = body.incline_pct
    return resp


@router.get("/bpm/status")
async def get_bpm_status(request: Request) -> BpmSyncStatus:
    state = get_state(request.app)
    controller = state.bpm_sync
    active = state.bpm_task is not None and not state.bpm_task.done()

    if controller is None:
        cfg = state.config
        return BpmSyncStatus(
            active=False,
            detected_bpm=None,
            selected_harmonic=None,
            effective_cadence=None,
            implied_stride_m=None,
            natural_stride_m=None,
            stride_score=None,
            commanded_speed_kmh=None,
            min_speed_kmh=cfg.bpm_min_speed_kmh,
            max_speed_kmh=cfg.bpm_max_speed_kmh,
        )

    last = controller.last_result
    return BpmSyncStatus(
        active=active,
        detected_bpm=last.detected_bpm if last else None,
        selected_harmonic=last.selected_harmonic if last else None,
        effective_cadence=last.effective_cadence if last else None,
        implied_stride_m=last.implied_stride_m if last else None,
        natural_stride_m=last.natural_stride_m if last else None,
        stride_score=last.stride_score if last else None,
        commanded_speed_kmh=last.speed_kmh if last else None,
        min_speed_kmh=controller.min_speed_kmh,
        max_speed_kmh=controller.max_speed_kmh,
    )


@router.put("/bpm/config")
async def update_bpm_config(
    request: Request, config: BpmSyncConfig
) -> BpmSyncConfig:
    """Update BPM sync parameters live."""
    state = get_state(request.app)

    if state.bpm_sync is None:
        state.bpm_sync = _make_controller(config, state.config)
    else:
        state.bpm_sync.min_speed_kmh = config.min_speed_kmh
        state.bpm_sync.max_speed_kmh = config.max_speed_kmh
        if config.harmonics is not None:
            state.bpm_sync.harmonics = tuple(config.harmonics)

    return config


@router.post("/bpm/harmonic/up")
async def harmonic_up(request: Request) -> dict[str, object]:
    """Shift to next higher harmonic (manual override)."""
    state = get_state(request.app)
    if state.bpm_sync is None:
        raise HTTPException(status_code=409, detail="BPM sync not active")
    h = state.bpm_sync.shift_harmonic(+1)
    return {"forced_harmonic": h, "harmonics": list(state.bpm_sync.harmonics)}


@router.post("/bpm/harmonic/down")
async def harmonic_down(request: Request) -> dict[str, object]:
    """Shift to next lower harmonic (manual override)."""
    state = get_state(request.app)
    if state.bpm_sync is None:
        raise HTTPException(status_code=409, detail="BPM sync not active")
    h = state.bpm_sync.shift_harmonic(-1)
    return {"forced_harmonic": h, "harmonics": list(state.bpm_sync.harmonics)}


@router.post("/bpm/harmonic/reset")
async def harmonic_reset(request: Request) -> dict[str, str]:
    """Clear harmonic override, return to auto."""
    state = get_state(request.app)
    if state.bpm_sync is None:
        raise HTTPException(status_code=409, detail="BPM sync not active")
    state.bpm_sync.reset_harmonic()
    return {"status": "auto"}


@router.post("/bpm/pause")
async def bpm_pause(request: Request) -> dict[str, str]:
    """Pause BPM speed control (keeps detecting, stops sending to treadmill)."""
    state = get_state(request.app)
    state.bpm_paused = True
    return {"status": "paused"}


@router.post("/bpm/resume")
async def bpm_resume(request: Request) -> dict[str, str]:
    """Resume BPM speed control."""
    state = get_state(request.app)
    state.bpm_paused = False
    return {"status": "resumed"}
