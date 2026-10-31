"""One inference session across the site, released after backend cleanup."""

from collections.abc import Awaitable, Callable

from fastapi import WebSocket

from voicelab_web.catalog import ModelId


class SessionLease:
    def __init__(self) -> None:
        self.active: ModelId | None = None
        self.owner: object | None = None

    def acquire(self, model: ModelId) -> object:
        if self.owner is not None:
            raise RuntimeError("Another model session is active")
        token = object()
        self.owner = token
        self.active = model
        return token

    def release(self, token: object) -> None:
        if self.owner is token:
            self.owner = None
            self.active = None

    async def run(
        self,
        socket: WebSocket,
        model: ModelId,
        operation: Callable[[], Awaitable[None]],
    ) -> None:
        origin = socket.headers.get("origin")
        expected = f"{'https' if socket.url.scheme == 'wss' else 'http'}://{socket.headers.get('host')}"
        if origin != expected:
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
        token = self.acquire(model)
        try:
            await operation()
        finally:
            self.release(token)
