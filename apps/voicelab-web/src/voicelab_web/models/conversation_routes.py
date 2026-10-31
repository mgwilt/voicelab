"""Compose local conversations under the same lease as local synthesis."""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, WebSocket

from voicelab_web.catalog_types import ModelId
from voicelab_web.models.ming_audio.session import serve_session as serve_ming
from voicelab_web.models.minicpm.session import serve_session as serve_minicpm
from voicelab_web.models.personaplex.session import serve_session as serve_personaplex
from voicelab_web.models.qwen_omni.session import serve_session as serve_qwen
from voicelab_web.transport.lease import SessionLease


def install(
    app: FastAPI,
    lease: SessionLease,
    release_local: Callable[[], Awaitable[None]],
) -> None:
    def add(
        model: ModelId,
        serve: Callable[[WebSocket], Awaitable[None]],
    ) -> None:
        async def endpoint(socket: WebSocket) -> None:
            async def run() -> None:
                await release_local()
                await serve(socket)

            await lease.run(socket, model, run)

        app.add_api_websocket_route(f"/api/{model}/session", endpoint)

    add("personaplex", serve_personaplex)
    add("ming_audio", serve_ming)
    add("qwen_omni", serve_qwen)
    add("minicpm", serve_minicpm)
