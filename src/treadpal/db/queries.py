from __future__ import annotations

from datetime import datetime

import aiosqlite

from treadpal.models import HistorySummary, TreadmillData


async def log_treadmill_data(db: aiosqlite.Connection, data: TreadmillData) -> None:
    await db.execute(
        """INSERT INTO treadmill_data
           (timestamp, speed_kmh, incline_pct, distance_m,
            elapsed_time_s, calories_kcal, heart_rate_bpm)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            data.timestamp.isoformat(),
            data.speed_kmh,
            data.incline_pct,
            data.distance_m,
            data.elapsed_time_s,
            data.calories_kcal,
            data.heart_rate_bpm,
        ),
    )
    await db.commit()


async def query_history(
    db: aiosqlite.Connection,
    start: datetime | None,
    end: datetime | None,
    limit: int,
    offset: int,
) -> list[TreadmillData]:
    conditions: list[str] = []
    params: list[object] = []
    if start:
        conditions.append("timestamp >= ?")
        params.append(start.isoformat())
    if end:
        conditions.append("timestamp <= ?")
        params.append(end.isoformat())

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"""SELECT * FROM treadmill_data {where}
              ORDER BY timestamp DESC LIMIT ? OFFSET ?"""
    params.extend([limit, offset])

    async with db.execute(sql, params) as cursor:
        rows = await cursor.fetchall()
        return [
            TreadmillData(
                timestamp=datetime.fromisoformat(row["timestamp"]),
                speed_kmh=row["speed_kmh"],
                incline_pct=row["incline_pct"],
                distance_m=row["distance_m"],
                elapsed_time_s=row["elapsed_time_s"],
                calories_kcal=row["calories_kcal"],
                heart_rate_bpm=row["heart_rate_bpm"],
            )
            for row in rows
        ]


async def get_session_summary(
    db: aiosqlite.Connection,
    start: datetime | None = None,
    end: datetime | None = None,
) -> HistorySummary | None:
    conditions: list[str] = []
    params: list[object] = []
    if start:
        conditions.append("timestamp >= ?")
        params.append(start.isoformat())
    if end:
        conditions.append("timestamp <= ?")
        params.append(end.isoformat())
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    sql = f"""SELECT
        MIN(timestamp) as session_start,
        MAX(timestamp) as session_end,
        MAX(elapsed_time_s) - MIN(elapsed_time_s) as duration_s,
        MAX(distance_m) as distance_m,
        AVG(speed_kmh) as avg_speed_kmh,
        MAX(speed_kmh) as max_speed_kmh,
        AVG(incline_pct) as avg_incline_pct,
        MAX(calories_kcal) as total_calories,
        AVG(heart_rate_bpm) as avg_heart_rate
    FROM treadmill_data {where}"""

    async with db.execute(sql, params) as cursor:
        row = await cursor.fetchone()
        if row is None or row["session_start"] is None:
            return None
        return HistorySummary(
            session_start=datetime.fromisoformat(row["session_start"]),
            session_end=datetime.fromisoformat(row["session_end"]),
            duration_s=row["duration_s"] or 0,
            distance_m=row["distance_m"] or 0,
            avg_speed_kmh=round(row["avg_speed_kmh"] or 0, 2),
            max_speed_kmh=round(row["max_speed_kmh"] or 0, 2),
            avg_incline_pct=round(row["avg_incline_pct"] or 0, 1),
            total_calories=row["total_calories"] or 0,
            avg_heart_rate=(
                round(row["avg_heart_rate"]) if row["avg_heart_rate"] else None
            ),
        )
