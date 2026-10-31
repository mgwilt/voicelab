"""One inference session across the site, released after backend cleanup."""

from collections.abc import Awaitable, Callable

from fastapi import WebSocket

from voicelab_web.catalog import ModelId


class SessionLease:
    def __init__(self) -> None:
        self.active: ModelId | None = None

    async def run(
        self,
        socket: WebSocket,
        model: ModelId,
        operation: Callable[[], Awaitable[None]],
    ) -> None:
        origin = socket.headers.get("origin")
        expected = f"{'https' if socket.url.scheme == 'wss' else 'http'}://{socket.headers.get('host')}"
        if origin is not None and origin != expected:
            await socket.close(code=1008)
            return
        if self.active is not None:
            await socket.accept()
            await socket.send_json(
                {
                    "type": "error",
                    "message": (
                        "Another model session is active. "
                        "Stop it before starting a new test."
                    ),
                }
            )
            await socket.close(code=1013)
            return
        # Runs on one event loop; no await between testing and acquiring ownership.
        self.active = model
        try:
            await operation()
        finally:
            self.active = None
