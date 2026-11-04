"""Run ASGI apps on real sockets (uvicorn in a thread) so streaming, disconnects and
backpressure behave as they do in production."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import uvicorn


class ServerThread:
    def __init__(self, app: Any, sndbuf: int | None = None) -> None:
        cfg = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error", lifespan="on")
        self.server = uvicorn.Server(cfg)
        self.server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        self.sock = socket.socket()
        if sndbuf is not None:  # accepted connections inherit it: lets tests fill buffers fast
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, sndbuf)
        self.sock.bind(("127.0.0.1", 0))
        self.thread = threading.Thread(
            target=self.server.run, kwargs={"sockets": [self.sock]}, daemon=True
        )

    def start(self) -> str:
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            if time.monotonic() > deadline or not self.thread.is_alive():
                raise RuntimeError("server failed to start")
            time.sleep(0.01)
        port = self.sock.getsockname()[1]
        return f"http://127.0.0.1:{port}"

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


@contextmanager
def serve(app: Any, sndbuf: int | None = None) -> Iterator[str]:
    srv = ServerThread(app, sndbuf)
    url = srv.start()
    try:
        yield url
    finally:
        srv.stop()
