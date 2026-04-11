from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import FastAPI

from treadpal.audio.bpm_sync import BpmSyncController
from treadpal.config import TreadPalConfig
from treadpal.db.database import init_database
from treadpal.models import TreadmillData

logger = logging.getLogger("treadpal")


class AppState:
    """Mutable runtime state shared across the application."""

    def __init__(self, config: TreadPalConfig, db: aiosqlite.Connection) -> None:
        self.config = config
        self.db = db
        # BLE
        self.ftms_client: object | None = None  # FTMSClient, set at runtime
        self.last_data: TreadmillData | None = None
        self.supported_features: list[str] = []
        # BPM sync
        self.bpm_sync: BpmSyncController | None = None
        self.bpm_task: asyncio.Task[None] | None = None
        self.bpm_paused: bool = False
        # Lock for connect/disconnect transitions
        self.lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = TreadPalConfig.load()
    db = await init_database(config.db_path)

    state = AppState(config=config, db=db)
    app.state.treadpal = state

    # Import here to avoid circular imports
    from treadpal.ble.scanner import run_scanner_loop

    scanner_task = asyncio.create_task(run_scanner_loop(state), name="ble-scanner")

    yield

    # Shutdown
    scanner_task.cancel()
    try:
        await scanner_task
    except asyncio.CancelledError:
        pass

    if state.bpm_task is not None:
        state.bpm_task.cancel()
        try:
            await state.bpm_task
        except asyncio.CancelledError:
            pass

    if state.ftms_client is not None:
        from treadpal.ble.ftms_client import FTMSClient

        assert isinstance(state.ftms_client, FTMSClient)
        await state.ftms_client.disconnect()

    await db.close()


def create_app() -> FastAPI:
    app = FastAPI(title="TreadPal", version="0.1.0", lifespan=lifespan)

    from treadpal.api.routes_audio import router as audio_router
    from treadpal.api.routes_bpm import router as bpm_router
    from treadpal.api.routes_control import router as control_router
    from treadpal.api.routes_history import router as history_router
    from treadpal.api.routes_status import router as status_router

    app.include_router(status_router, prefix="/api")
    app.include_router(control_router, prefix="/api")
    app.include_router(history_router, prefix="/api")
    app.include_router(bpm_router, prefix="/api")
    app.include_router(audio_router)

    return app


def get_state(app: FastAPI) -> AppState:
    """Helper to retrieve AppState from a FastAPI app."""
    return app.state.treadpal  # type: ignore[no-any-return]
