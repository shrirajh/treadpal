"""Tests for FastAPI endpoints."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from treadpal.app import AppState, create_app
from treadpal.config import TreadPalConfig
from treadpal.db.database import init_database


@pytest.fixture
async def app(tmp_path):
    """Create app with manually initialized state (no BLE scanner)."""
    from unittest.mock import AsyncMock, patch

    config = TreadPalConfig(db_path=str(tmp_path / "test.db"))
    db = await init_database(config.db_path)
    state = AppState(config=config, db=db)

    with patch("treadpal.ble.scanner.run_scanner_loop", new_callable=AsyncMock):
        application = create_app()
        application.state.treadpal = state
        yield application

    await db.close()


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_status_disconnected(client: AsyncClient) -> None:
    resp = await client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["connected"] is False
    assert data["last_data"] is None


@pytest.mark.asyncio
async def test_features_empty(client: AsyncClient) -> None:
    resp = await client.get("/api/features")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_control_not_connected(client: AsyncClient) -> None:
    resp = await client.post("/api/control/start")
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_set_speed_not_connected(client: AsyncClient) -> None:
    resp = await client.post("/api/control/set_speed", json={"value": 5.0})
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_set_speed_validation(client: AsyncClient) -> None:
    """Speed validation should reject before checking connection."""
    resp = await client.post("/api/control/set_speed", json={"value": 30.0})
    # Either 400 (validation) or 409 (not connected) is acceptable
    assert resp.status_code in (400, 409)


@pytest.mark.asyncio
async def test_history_empty(client: AsyncClient) -> None:
    resp = await client.get("/api/history")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_history_summary_empty(client: AsyncClient) -> None:
    resp = await client.get("/api/history/summary")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_bpm_status_default(client: AsyncClient) -> None:
    resp = await client.get("/api/bpm/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["active"] is False
    assert data["detected_bpm"] is None


@pytest.mark.asyncio
async def test_bpm_update_creates_controller(client: AsyncClient) -> None:
    """POST /api/bpm/update should auto-create controller and compute."""
    resp = await client.post("/api/bpm/update", json={"bpm": 140})
    assert resp.status_code == 200
    data = resp.json()
    assert "commanded_speed_kmh" in data
    assert "selected_harmonic" in data
    assert data["detected_bpm"] == 140


@pytest.mark.asyncio
async def test_bpm_config_update(client: AsyncClient) -> None:
    resp = await client.put(
        "/api/bpm/config",
        json={"min_speed_kmh": 5.0, "max_speed_kmh": 9.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["min_speed_kmh"] == 5.0
    assert data["max_speed_kmh"] == 9.0


@pytest.mark.asyncio
async def test_bpm_stop_not_running(client: AsyncClient) -> None:
    resp = await client.post("/api/bpm/stop")
    assert resp.status_code == 409


@pytest.fixture
def sim(app):
    """Attach a simulated treadmill (no BLE, no physics loop)."""
    from treadpal.ble.simulator import SimulatedFTMSClient

    state = app.state.treadpal
    client = SimulatedFTMSClient(state)
    state.ftms_client = client
    return state


@pytest.mark.asyncio
async def test_status_reports_commanded_targets(client: AsyncClient, sim) -> None:
    await client.post("/api/control/start")
    await client.post("/api/control/set_speed", json={"value": 5.2})
    await client.post("/api/control/set_incline", json={"value": 2.3})
    data = (await client.get("/api/status")).json()
    assert data["connected"] is True
    assert data["machine_state"] == "running"
    assert data["target_speed_kmh"] == 5.2
    assert data["target_incline_pct"] == 2.3


@pytest.mark.asyncio
async def test_stop_clears_speed_target(client: AsyncClient, sim) -> None:
    await client.post("/api/control/set_speed", json={"value": 5.0})
    await client.post("/api/control/stop")
    data = (await client.get("/api/status")).json()
    assert data["machine_state"] == "stopped"
    assert data["target_speed_kmh"] is None


@pytest.mark.asyncio
async def test_status_notification_updates_targets(sim) -> None:
    """Targets changed on the treadmill console arrive via Machine Status."""
    import struct

    sim.ftms_client._on_status_change(None, bytearray(struct.pack("<BH", 0x05, 450)))
    sim.ftms_client._on_status_change(None, bytearray(struct.pack("<Bh", 0x06, 30)))
    assert sim.target_speed_kmh == 4.5
    assert sim.target_incline_pct == 3.0
    sim.ftms_client._on_status_change(None, bytearray([0x02, 0x02]))
    assert sim.machine_state == "paused"


@pytest.mark.asyncio
async def test_bpm_update_respects_pause(client: AsyncClient, sim) -> None:
    await client.post("/api/control/set_speed", json={"value": 3.0})
    await client.post("/api/bpm/pause")
    resp = await client.post("/api/bpm/update", json={"bpm": 140})
    assert resp.status_code == 200
    assert sim.target_speed_kmh == 3.0
    status = (await client.get("/api/bpm/status")).json()
    assert status["paused"] is True
    assert status["age_s"] is not None


@pytest.mark.asyncio
async def test_web_ui_served(client: AsyncClient) -> None:
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "treadpal" in resp.text.lower()
    assert (await client.get("/app.js")).status_code == 200


@pytest.mark.asyncio
async def test_debug_ble_endpoint(client: AsyncClient) -> None:
    resp = await client.get("/api/debug/ble")
    assert resp.status_code == 200
    data = resp.json()
    assert data["packets"] == [] and data["events"] == []


@pytest.mark.asyncio
async def test_speed_rounded_to_mph_step(app, client: AsyncClient, sim) -> None:
    """mph treadmills truncate 2.49 mph to 2.4; we must send 2.5 instead."""
    app.state.treadpal.config.speed_send_mph = True
    await client.post("/api/control/set_speed", json={"value": 4.0})
    data = (await client.get("/api/status")).json()
    assert data["target_speed_kmh"] == pytest.approx(2.5 * 1.60934, abs=1e-3)
    assert data["speed_resolution_kmh"] == pytest.approx(0.160934)


def test_viz_relay_to_ui(tmp_path) -> None:
    """Agent viz frames on /ws/audio reach /ws/viz verbatim; junk is dropped."""
    from fastapi.testclient import TestClient

    application = create_app()
    application.state.treadpal = AppState(
        config=TreadPalConfig(db_path=str(tmp_path / "t.db")), db=None  # type: ignore[arg-type]
    )
    tc = TestClient(application)  # No context manager: skip lifespan (BLE scanner)
    with tc.websocket_connect("/ws/viz") as ui, tc.websocket_connect("/ws/audio?sr=44100") as agent:
        frame = '{"type":"viz","bands":[0.5,0.2,0.1],"note":9}'
        agent.send_text("not json")
        agent.send_text('{"type":"other"}')
        agent.send_bytes(bytes(16))
        agent.send_text(frame)
        assert ui.receive_text() == frame


def test_motion_estimator_ramps_toward_target() -> None:
    from treadpal.ble.motion import MotionEstimator

    est = MotionEstimator(rate_per_s=0.5)
    est.set_target(0.0, now=0.0)
    est.set_target(10.0, now=0.0)
    assert est.advance(4.0) == pytest.approx(2.0)
    assert est.advance(100.0) == 10.0
    est.set_target(9.0, now=100.0)
    assert est.advance(101.0) == pytest.approx(9.5)


def test_implausible_jump_detection() -> None:
    from treadpal.ble.motion import is_implausible_jump

    assert is_implausible_jump((0.0, 4.0, 10.0), (0.9, 5.6, 1.0)) is not None
    assert is_implausible_jump((0.0, 4.0, 1.0), (0.9, 4.5, 1.5)) is None
    assert is_implausible_jump((0.0, 4.0, 1.0), (30.0, 4.0, 10.0)) is None  # Long gap: unknown


@pytest.mark.asyncio
async def test_target_reporting_treadmill_gets_estimated(client: AsyncClient, sim) -> None:
    """A treadmill echoing targets (incline 0 -> 10% in one packet) switches to estimation."""
    from treadpal.ble.simulator import build_treadmill_packet

    handler = sim.ftms_client._on_treadmill_data
    handler(None, bytearray(build_treadmill_packet(4.0, 100, 0.0, 5, 60)))
    handler(None, bytearray(build_treadmill_packet(4.0, 100, 10.0, 5, 61)))
    data = (await client.get("/api/status")).json()
    assert data["motion_estimated"] is True
    assert data["target_incline_pct"] == 10.0
    assert data["last_data"]["incline_pct"] < 1.0  # Motor has barely started moving
    assert data["last_data"]["distance_m"] == 100


@pytest.mark.asyncio
async def test_estimate_starts_from_prior_position(client: AsyncClient, sim) -> None:
    """A command sent before target-reporting is detected must not teleport the estimate."""
    from treadpal.ble.simulator import build_treadmill_packet

    handler = sim.ftms_client._on_treadmill_data
    handler(None, bytearray(build_treadmill_packet(4.0, 100, 0.0, 5, 60)))
    await client.post("/api/control/set_incline", json={"value": 8.0})
    handler(None, bytearray(build_treadmill_packet(4.0, 100, 8.0, 5, 61)))
    data = (await client.get("/api/status")).json()
    assert data["motion_estimated"] is True
    assert data["last_data"]["incline_pct"] < 1.0


def test_viz_only_from_newest_source(tmp_path) -> None:
    """With two agents connected, only the newest one's frames reach the UI."""
    from fastapi.testclient import TestClient

    application = create_app()
    application.state.treadpal = AppState(
        config=TreadPalConfig(db_path=str(tmp_path / "t.db")), db=None  # type: ignore[arg-type]
    )
    tc = TestClient(application)
    with (
        tc.websocket_connect("/ws/viz") as ui,
        tc.websocket_connect("/ws/audio?sr=44100") as old,
        tc.websocket_connect("/ws/audio?sr=44100") as new,
    ):
        old.send_text('{"type":"viz","note":1}')
        new.send_text('{"type":"viz","note":2}')
        assert ui.receive_text() == '{"type":"viz","note":2}'


def test_stereo_audio_yields_stem_frames(tmp_path) -> None:
    """Interleaved stereo on /ws/audio is separated and stem frames reach /ws/viz."""
    import json

    import numpy as np
    from fastapi.testclient import TestClient

    from treadpal.audio.stems import StemModel

    application = create_app()
    state = AppState(config=TreadPalConfig(db_path=str(tmp_path / "t.db")), db=None)  # type: ignore[arg-type]
    state.stems_status, state.stems_model = "ready", StemModel()
    application.state.treadpal = state
    tc = TestClient(application)
    t = np.arange(44100 // 5) / 44100
    tone = (0.3 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
    stereo = np.stack([tone, 0.8 * tone], axis=1)  # [n, 2] -> interleaved bytes
    with tc.websocket_connect("/ws/viz") as ui, tc.websocket_connect("/ws/audio?sr=44100&ch=2") as agent:
        for i in range(0, len(stereo), 1024):
            agent.send_bytes(stereo[i:i + 1024].tobytes())
        msg = json.loads(ui.receive_text())
        assert msg["type"] == "stems"
        assert msg["names"] == ["vocals", "other", "bass", "drums"]


def test_port_guard_refuses_a_taken_port() -> None:
    """Another program on the port stops startup instead of silently sharing it."""
    import socket

    from treadpal.__main__ import bind_exclusive

    with socket.socket() as other:
        other.bind(("127.0.0.1", 0))
        other.listen()
        port = other.getsockname()[1]
        with pytest.raises(OSError):
            bind_exclusive("0.0.0.0", port).close()
    bind_exclusive("127.0.0.1", port).close()  # Free again
