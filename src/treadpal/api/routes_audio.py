"""WebSocket endpoint for streaming audio from remote clients.

Protocol:
  1. Client connects to /ws/audio?sr=48000
  2. Client sends binary frames of mono float32 PCM audio
  3. Server accumulates in buffer, runs beat_this every N seconds
  4. Server sends JSON status messages back: {"bpm": 140, "speed_kmh": 5.5, ...}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from treadpal.app import get_state
from treadpal.audio.beat_detector import AudioBuffer, detect_bpm

logger = logging.getLogger("treadpal.audio.ws")

router = APIRouter()


@router.websocket("/ws/audio")
async def audio_websocket(ws: WebSocket, sr: int = Query(default=44100)) -> None:
    """Accept streaming audio, detect BPM, control treadmill."""
    await ws.accept()
    state = get_state(ws.app)
    cfg = state.config

    buf = AudioBuffer(sr=sr, max_seconds=12.0)
    analysis_hz = 5.0  # Target analysis rate in Hz
    window_seconds = 8.0
    warmup_seconds = 5.0

    logger.info("Audio WebSocket connected (sr=%d)", sr)

    # Speed ramper state
    current_speed: float | None = None
    max_ramp = 0.1

    # Incline state
    last_incline_time = 0.0
    incline_interval = 30.0

    async def _analyze_loop() -> None:
        nonlocal current_speed, last_incline_time

        # Wait for warmup
        while buf.seconds_available < warmup_seconds:
            await asyncio.sleep(0.5)
        logger.info("Audio warmup complete, starting beat detection")

        while True:
            t_start = time.monotonic()
            audio = buf.get_last(window_seconds)
            if audio is None:
                logger.debug("Not enough audio yet (%.1fs)", buf.seconds_available)
                await asyncio.sleep(1.0)
                continue

            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < 0.001:
                logger.debug("Silence (rms=%.6f)", rms)
                await asyncio.sleep(1.0 / analysis_hz)
                continue

            logger.debug("Analyzing %.1fs audio (rms=%.4f)...", window_seconds, rms)

            # Run beat detection in a thread (blocks until complete)
            loop = asyncio.get_event_loop()
            bpm = await loop.run_in_executor(None, detect_bpm, audio, sr)

            logger.debug("Analysis complete: bpm=%s", bpm)

            if bpm is None:
                logger.info("No beats detected (rms=%.4f)", rms)
                continue
            if bpm < 40 or bpm > 220:
                logger.info("BPM out of range: %.1f", bpm)
                continue

            # Compute target speed via harmonic snapper
            if state.bpm_sync is None:
                from treadpal.audio.bpm_sync import BpmSyncController
                state.bpm_sync = BpmSyncController(
                    min_speed_kmh=cfg.bpm_min_speed_kmh,
                    max_speed_kmh=cfg.bpm_max_speed_kmh,
                    harmonics=cfg.bpm_harmonics,
                    speed_step_kmh=cfg.speed_step_kmh,
                )

            result = state.bpm_sync.compute(bpm)
            target_speed = result.speed_kmh

            # Ramp toward target
            if current_speed is None:
                current_speed = target_speed
            else:
                diff = target_speed - current_speed
                if abs(diff) <= max_ramp:
                    current_speed = target_speed
                else:
                    current_speed += max_ramp if diff > 0 else -max_ramp
                current_speed = round(current_speed, 2)

            # Send speed to treadmill
            if state.ftms_client is not None:
                from treadpal.ble.ftms_client import FTMSClient
                assert isinstance(state.ftms_client, FTMSClient)
                if state.ftms_client.is_connected:
                    await state.ftms_client.set_target_speed(current_speed)
                    logger.info("BPM=%.1f -> %.2f km/h (ramp=%.2f, x%.2f)",
                                bpm, target_speed, current_speed, result.selected_harmonic)

            # Incline from energy (throttled, only if configured)
            incline: float | None = None
            if cfg.home_incline is not None:
                now = time.monotonic()
                if (now - last_incline_time) >= incline_interval:
                    short = buf.get_last(3.0)
                    if short is not None:
                        from treadpal.audio.intensity import compute_intensity_incline
                        incline = compute_intensity_incline(short, sr, state)
                        if incline is not None and state.ftms_client is not None:
                            from treadpal.ble.ftms_client import FTMSClient
                            assert isinstance(state.ftms_client, FTMSClient)
                            if state.ftms_client.is_connected:
                                await state.ftms_client.set_target_incline(incline)
                        last_incline_time = now

            # Send status back to client
            status = {
                "bpm": bpm,
                "target_speed_kmh": target_speed,
                "ramped_speed_kmh": current_speed,
                "harmonic": result.selected_harmonic,
                "stride_m": result.implied_stride_m,
            }
            if incline is not None:
                status["incline_pct"] = incline

            try:
                await ws.send_json(status)
            except (WebSocketDisconnect, RuntimeError):
                break

            # Throttle to target Hz (accounts for analysis time)
            elapsed = time.monotonic() - t_start
            target_period = 1.0 / analysis_hz
            if elapsed < target_period:
                await asyncio.sleep(target_period - elapsed)

    analysis_task = asyncio.create_task(_analyze_loop(), name="ws-beat-detect")

    try:
        while True:
            data = await ws.receive_bytes()
            samples = np.frombuffer(data, dtype=np.float32)
            buf.append(samples)
    except WebSocketDisconnect:
        logger.info("Audio WebSocket disconnected")
    finally:
        analysis_task.cancel()
        try:
            await analysis_task
        except asyncio.CancelledError:
            pass
