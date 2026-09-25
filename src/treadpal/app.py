from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from treadpal.audio.bpm_sync import BpmSyncController
from treadpal.config import TreadPalConfig
from treadpal.db.database import init_database
from treadpal.models import TreadmillData, ValueRange
from treadpal.viz import VizHub

if TYPE_CHECKING:
    from treadpal.audio.stems import StemModel

logger = logging.getLogger("treadpal")

_WEB_DIR = Path(__file__).parent / "web"


class AppState:
    """Mutable runtime state shared across the application."""

    def __init__(self, config: TreadPalConfig, db: aiosqlite.Connection) -> None:
        self.config = config
        self.db = db
        # BLE
        self.ftms_client: object | None = None  # FTMSClient, set at runtime
        self.last_data: TreadmillData | None = None
        self.supported_features: list[str] = []
        self.target_speed_kmh: float | None = None
        self.target_incline_pct: float | None = None
        self.machine_state: str | None = None
        self.motion_estimated: bool = False  # Speed/incline are modelled, not reported
        self.speed_range: ValueRange | None = None
        self.incline_range: ValueRange | None = None
        # Recent raw BLE traffic for /api/debug/ble
        self.ble_packets: deque[dict[str, Any]] = deque(maxlen=40)
        self.ble_events: deque[dict[str, Any]] = deque(maxlen=80)
        # BPM sync
        self.bpm_sync: BpmSyncController | None = None
        self.bpm_task: asyncio.Task[None] | None = None
        self.bpm_paused: bool = False
        self.audio_clients: int = 0  # Connected /ws/audio streamers
        self.bpm_updated_at: float | None = None  # time.monotonic() of last BPM result
        self.viz = VizHub()  # Live spectrum/beat data for web UI clients
        self.viz_source: object | None = None  # Audio connection whose data reaches the UI
        # Stem separation: "off", "preparing", "ready" or "error"
        self.stems_status: str = "off"
        self.stems_model: StemModel | None = None
        # Lock for connect/disconnect transitions
        self.lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = TreadPalConfig.load()
    db = await init_database(config.db_path)

    state = AppState(config=config, db=db)
    app.state.treadpal = state

    # Import here to avoid circular imports
    if config.simulate:
        from treadpal.ble.simulator import run_simulator

        scanner_task = asyncio.create_task(run_simulator(state), name="simulator")
    else:
        from treadpal.ble.scanner import run_scanner_loop

        scanner_task = asyncio.create_task(
            run_scanner_loop(state), name="ble-scanner"
        )

    stems_task = asyncio.create_task(_prepare_stems(state), name="stems-prepare") if config.stems else None
    beat_task = asyncio.create_task(_preload_beat_model(), name="beat-preload")

    yield

    # Shutdown
    if stems_task is not None:
        stems_task.cancel()
    beat_task.cancel()
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


async def _prepare_stems(state: AppState) -> None:
    """Load (and warm up) the vendored stem model off the event loop; audio works without it."""
    from treadpal.audio.stems import StemModel, verified_model

    state.stems_status = "preparing"
    try:
        state.stems_model = await asyncio.to_thread(
            lambda: StemModel(verified_model(), state.config.stems_threads)
        )
        state.stems_status = "ready"
        logger.info("Stem separation ready")
    except Exception:
        state.stems_status = "error"
        logger.exception("Stem separation unavailable")


async def _preload_beat_model() -> None:
    """Load beat_this now: loading it at the first detection stalls live audio for seconds."""
    from treadpal.audio.beat_detector import preload

    try:
        await asyncio.to_thread(preload)
    except Exception:
        logger.exception("Could not preload the beat detector")


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

    # Web UI (mounted last so it doesn't shadow API routes)
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")

    return app


def get_state(app: FastAPI) -> AppState:
    """Helper to retrieve AppState from a FastAPI app."""
    return app.state.treadpal  # type: ignore[no-any-return]
