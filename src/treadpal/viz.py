"""Fan-out of live music data (spectrum frames, beat grid) to web UI clients."""

from __future__ import annotations

import asyncio
import json
from typing import Any


class VizHub:
    """Each subscriber gets a small queue; slow clients drop old frames, never block."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[str]] = set()

    def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=8)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        self._subscribers.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def publish_text(self, text: str) -> None:
        for q in self._subscribers:
            if q.full():
                q.get_nowait()
            q.put_nowait(text)

    def publish(self, message: dict[str, Any]) -> None:
        if self._subscribers:
            self.publish_text(json.dumps(message, separators=(",", ":")))
