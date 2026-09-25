"""Entry point: `uv run treadpal` or `python -m treadpal`."""

from __future__ import annotations

import logging
import socket
import sys

import uvicorn

from treadpal.config import TreadPalConfig

logger = logging.getLogger("treadpal")


def _someone_listening(port: int) -> str | None:
    """Loopback address where another program already answers on this port, if any."""
    for family, addr in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.settimeout(0.3)
                if s.connect_ex((addr, port)) == 0:
                    return addr
        except OSError:
            continue  # e.g. no IPv6
    return None


def bind_exclusive(host: str, port: int) -> socket.socket:
    """Bind the listening socket so no other program can share the port.

    uvicorn binds with SO_REUSEADDR, which on Windows lets a second program
    (e.g. a Docker container publishing the same port) bind it too; the OS then
    splits connections between the two, so pages and websockets land on the
    wrong server at random.
    """
    taken = _someone_listening(port)
    if taken is not None:
        raise OSError(f"another program is already listening on {taken} port {port}")
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    if sys.platform == "win32":
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        sock.close()
        raise
    return sock


def main() -> None:
    config = TreadPalConfig.load()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Only our loggers go to debug; bleak/uvicorn debug output is overwhelming
    logging.getLogger("treadpal").setLevel(config.log_level.upper())

    try:
        sock = bind_exclusive(config.host, config.port)
    except OSError as e:
        logger.error(
            "Can't use port %d: %s. Stop that program, or run TreadPal on another port "
            "(TREADPAL_PORT=8090, or \"port\" in treadpal.json).", config.port, e,
        )
        sys.exit(1)

    server = uvicorn.Server(uvicorn.Config("treadpal.app:create_app", factory=True, log_level="info"))
    server.run(sockets=[sock])


if __name__ == "__main__":
    main()
