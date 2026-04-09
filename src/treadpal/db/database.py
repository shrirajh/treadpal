from __future__ import annotations

import aiosqlite

from treadpal.db.schema import SCHEMA_SQL


async def init_database(db_path: str) -> aiosqlite.Connection:
    """Open database, create tables if needed, return connection."""
    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    await db.executescript(SCHEMA_SQL)
    await db.commit()
    return db
