"""Compose typed synthesis routes and release inactive resident models."""

import asyncio
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Protocol

from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

from voicelab.conversations.contracts import Voice
from voicelab_web.catalog_types import ModelId
from voicelab_web.conversations.adapters.synthesis import SynthesisVoice
from voicelab_web.models.synthesis.service import SynthesisService
from voicelab_web.models.synthesis.session import serve_session
from voicelab_web.transport.lease import SessionLease


class Unloadable(Protocol):
    def unload(self) -> None: ...


class SynthesisRoutes:
    def __init__(
        self,
        app: FastAPI,
        lease: SessionLease,
        release_duplex: Callable[[], Awaitable[None]],
    ) -> None:
        self.app = app
        self.lease = lease
        self.release_duplex = release_duplex
        self.services: list[Unloadable] = []
        self.voices: dict[ModelId, Callable[[BaseModel], Voice]] = {}

    async def release(self, keep: Unloadable | None = None) -> None:
        for service in self.services:
            if service is not keep:
                await asyncio.to_thread(service.unload)

    def add[Request: BaseModel](
        self, model: ModelId, service: SynthesisService[Request]
    ) -> None:
        self.services.append(service)
        self.voices[model] = lambda settings: SynthesisVoice(service, settings)

        async def run(socket: WebSocket) -> None:
            await self.release_duplex()
            await self.release(service)
            await serve_session(socket, service)

        async def endpoint(socket: WebSocket) -> None:
            await self.lease.run(socket, model, partial(run, socket))

        self.app.add_api_websocket_route(f"/api/{model}/session", endpoint)
