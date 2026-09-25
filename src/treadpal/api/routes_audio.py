"""WebSocket endpoint for streaming audio from remote clients.

Protocol:
  1. Client connects to /ws/audio?sr=48000&ch=2
  2. Client sends binary frames of float32 PCM audio (interleaved if ch > 1)
  3. Server accumulates in buffer, runs beat_this every N seconds
  4. Server sends JSON status messages back: {"bpm": 140, "speed_kmh": 5.5, ...}
  5. Client may also send text frames of JSON visualiser data ({"type": "viz", ...});
     these are relayed as-is to web UI clients on /ws/viz, along with the beat grid.
  6. When stem separation is ready, the server splits the audio into vocals,
     other, bass and drums and publishes {"type": "stems", ...} frames too.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from treadpal.app import get_state
from treadpal.audio.beat_detector import AudioBuffer, detect_beats
from treadpal.audio.stems import StemWorker

logger = logging.getLogger("treadpal.audio.ws")

router = APIRouter()


@router.websocket("/ws/audio")
async def audio_websocket(
    ws: WebSocket, sr: int = Query(default=44100), ch: int = Query(default=1, ge=1, le=8)
) -> None:
    """Accept streaming audio, detect BPM, control treadmill."""
    await ws.accept()
    state = get_state(ws.app)
    cfg = state.config

    buf = AudioBuffer(sr=sr, max_seconds=12.0)
    # Target analysis rate in Hz. Each run looks at 8 s of audio; the UI's beat
    # clock runs between results, so faster only costs CPU
    analysis_hz = 2.5
    window_seconds = 8.0
    warmup_seconds = 5.0

    logger.info("Audio WebSocket connected (sr=%d, ch=%d)", sr, ch)

    # Speed ramper state
    current_speed: float | None = None
    max_ramp = 0.5 / analysis_hz  # km/h per analysis: 0.5 km/h per second

    # Incline state
    last_incline_time = 0.0
    incline_interval = 30.0

    last_sent_speed: float | None = None

    # Server clock time the newest buffered sample arrived; anchors beat times
    last_arrival = time.monotonic()

    async def _analyze_loop() -> None:
        nonlocal current_speed, last_incline_time, last_sent_speed

        # Wait for warmup
        while buf.seconds_available < warmup_seconds:
            await asyncio.sleep(0.5)
        logger.info("Audio warmup complete, starting beat detection")

        while True:
            t_start = time.monotonic()
            window_end_at = last_arrival
            # No audio arriving (music stopped: WASAPI loopback sends nothing in
            # silence, or the agent stalled): the buffer is history, not music
            if t_start - last_arrival > 1.0:
                await asyncio.sleep(0.25)
                continue
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
            detected = await loop.run_in_executor(None, detect_beats, audio, sr)

            logger.debug("Analysis complete: %s", detected)

            if detected is None:
                logger.info("No beats detected (rms=%.4f)", rms)
                continue
            bpm, grid = detected
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
            state.bpm_updated_at = time.monotonic()

            # Beat phase for the UI, as ages ("last beat was N s ago") so the
            # browser needs no clock sync, only the (small) delivery latency
            if grid is not None:
                now = time.monotonic()
                window_start_at = window_end_at - window_seconds
                beat_msg: dict[str, object] = {
                    "type": "beat",
                    "bpm": bpm,
                    "period": round(grid.period_s, 4),
                    "age": round(now - (window_start_at + grid.last_beat_s), 4),
                    "conf": grid.confidence,
                    "harmonic": result.selected_harmonic,
                }
                if grid.bar_length and grid.last_downbeat_s is not None:
                    beat_msg["bar"] = grid.bar_length
                    beat_msg["down_age"] = round(now - (window_start_at + grid.last_downbeat_s), 4)
                if state.viz_source == source:
                    state.viz.publish(beat_msg)
            target_speed = result.speed_kmh

            # While paused, follow the belt so resuming ramps from the actual speed
            if state.bpm_paused:
                last_sent_speed = None
                if state.last_data is not None:
                    current_speed = state.last_data.speed_kmh

            # Check if user manually changed speed on the treadmill
            # Compare against what we last SENT, not our ramp target
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

            # Send speed to treadmill (skip if paused). Only on a real change: the
            # loop runs at 5 Hz, each command is 2 BLE writes, and BPM jitter near a
            # treadmill step boundary (0.1 mph) would otherwise flip-flop the belt
            res = cfg.speed_resolution_kmh
            step_speed = round(current_speed / res) * res if current_speed is not None else None
            if (
                step_speed is not None
                and (last_sent_speed is None or abs(current_speed - last_sent_speed) > 0.75 * res)
                and not state.bpm_paused
                and state.ftms_client is not None
            ):
                from treadpal.ble.ftms_client import FTMSClient
                assert isinstance(state.ftms_client, FTMSClient)
                if state.ftms_client.is_connected:
                    await state.ftms_client.set_target_speed(step_speed)
                    last_sent_speed = step_speed
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
                "harmonic_override": result.harmonic_override,
                "stride_m": result.implied_stride_m,
                "paused": state.bpm_paused,
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
    state.audio_clients += 1
    # The newest source drives the UI's visuals; two interleaved beat grids would fight
    source = object()
    state.viz_source = source

    loop = asyncio.get_running_loop()
    stems: StemWorker | None = None

    def emit_stems(frame: dict[str, object]) -> None:  # Called on the separator thread
        loop.call_soon_threadsafe(_publish_stems, frame)

    def _publish_stems(frame: dict[str, object]) -> None:
        if state.viz_source == source:
            state.viz.publish(frame)

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                samples = np.frombuffer(msg["bytes"], dtype=np.float32)
                if ch > 1:
                    frames = samples[: len(samples) // ch * ch].reshape(-1, ch)
                    buf.append(frames.mean(axis=1))
                    stereo = frames[:, :2].T
                else:
                    buf.append(samples)
                    stereo = np.stack([samples, samples])
                last_arrival = time.monotonic()
                if stems is None and state.stems_status == "ready" and state.stems_model is not None:
                    try:
                        stems = StemWorker(state.stems_model, sr, emit_stems, cfg.stems_floor_db)
                        logger.info("Stem separation started for this stream")
                    except Exception:
                        logger.exception("Could not start stem separation")
                        state.stems_status = "error"
                if stems is not None and state.viz_source == source:
                    stems.feed(np.ascontiguousarray(stereo))
            elif msg.get("text"):
                if state.viz_source == source:
                    _relay_viz(state, msg["text"])
    except WebSocketDisconnect:
        pass
    finally:
        logger.info("Audio WebSocket disconnected")
        if stems is not None:
            stems.close()
            logger.info("Stem separation used %.0f%% of real time (%d blocks skipped)",
                        stems.load * 100, stems.dropped)
        state.audio_clients -= 1
        if state.viz_source == source:
            state.viz_source = None
        if state.audio_clients == 0:
            state.viz.publish({"type": "viz_end"})
        analysis_task.cancel()
        try:
            await analysis_task
        except asyncio.CancelledError:
            pass


def _relay_viz(state: object, text: str) -> None:
    """Forward an agent's visualiser frame to UI clients, if it looks like one."""
    from treadpal.app import AppState

    assert isinstance(state, AppState)
    if not state.viz.subscriber_count:
        return
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        return
    if isinstance(msg, dict) and msg.get("type") == "viz":
        state.viz.publish_text(text)


@router.websocket("/ws/viz")
async def viz_websocket(ws: WebSocket) -> None:
    """Push live music data to the web UI: agent spectrum frames and the beat grid."""
    await ws.accept()
    state = get_state(ws.app)
    q = state.viz.subscribe()
    try:
        while True:
            try:
                text = await asyncio.wait_for(q.get(), timeout=5.0)
            except TimeoutError:
                text = '{"type":"idle"}'  # Keepalive; also detects dead clients
            await ws.send_text(text)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        state.viz.unsubscribe(q)
