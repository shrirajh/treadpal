from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Request

from treadpal.app import get_state
from treadpal.db.queries import get_session_summary, query_history
from treadpal.models import HistorySummary, TreadmillData

router = APIRouter(tags=["history"])


@router.get("/history")
async def get_history(
    request: Request,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 1000,
    offset: int = 0,
) -> list[TreadmillData]:
    state = get_state(request.app)
    return await query_history(state.db, start, end, limit, offset)


@router.get("/history/summary")
async def get_summary(
    request: Request,
    start: datetime | None = None,
    end: datetime | None = None,
) -> HistorySummary | None:
    state = get_state(request.app)
    return await get_session_summary(state.db, start, end)
