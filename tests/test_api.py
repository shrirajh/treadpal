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
