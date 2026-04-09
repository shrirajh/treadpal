"""Entry point: `uv run treadpal` or `python -m treadpal`."""

from __future__ import annotations

import logging

import uvicorn

from treadpal.config import TreadPalConfig


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = TreadPalConfig.load()

    uvicorn.run(
        "treadpal.app:create_app",
        factory=True,
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
