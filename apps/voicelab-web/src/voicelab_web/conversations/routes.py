"""Same-origin conversation reservations, signaling, and explicit teardown."""

import asyncio
import platform
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from voicelab.conversations.contracts import ConversationSpec
from voicelab_web.conversations.requests import REQUEST, ConversationRequest
from voicelab_web.conversations.session import ConversationSession
from voicelab_web.conversations.transport.schemas import (
    Answer,
    Created,
    IceUpdate,
    Offer,
)
from voicelab_web.transport.lease import SessionLease


@dataclass
class Reservation:
    request: ConversationRequest
    token: object
    session: ConversationSession | None = None
    task: asyncio.Task[None] | None = None
    expiry: asyncio.TimerHandle | None = None


class Conversations:
    def __init__(
        self,
        lease: SessionLease,
        release_local: Callable[[], Awaitable[None]],
        factory: Callable[[ConversationRequest], ConversationSpec],
    ) -> None:
        self.lease = lease
        self.release_local = release_local
        self.factory = factory
        self.sessions: dict[str, Reservation] = {}

    def create(self, request: ConversationRequest) -> Created:
        token = self.lease.acquire(request.model)
        identity = secrets.token_urlsafe(24)
        reservation = Reservation(request, token)
        self.sessions[identity] = reservation

        def expire() -> None:
            if reservation.session is None:
                self.sessions.pop(identity, None)
                self.lease.release(token)

        reservation.expiry = asyncio.get_running_loop().call_later(30, expire)
        return Created(
            session_id=identity, offer_url=f"/api/conversations/{identity}/offer"
        )

    def get(self, identity: str) -> Reservation:
        reservation = self.sessions.get(identity)
        if reservation is None:
            raise HTTPException(404, "Conversation expired or ended")
        return reservation

    async def offer(self, identity: str, offer: Offer) -> Answer:
        reservation = self.get(identity)
        if reservation.session is not None:
            raise HTTPException(409, "Conversation already has a peer")
        if reservation.expiry:
            reservation.expiry.cancel()
        try:
            # Mark construction with the session before awaiting signaling.
            session = ConversationSession(self.factory(reservation.request))
            reservation.session = session
            answer = await session.offer(offer)
            if self.sessions.get(identity) is not reservation:
                raise HTTPException(410, "Conversation ended during signaling")
            reservation.task = asyncio.create_task(
                self._run(identity, reservation, session)
            )
            return answer
        except BaseException:
            await self.end(identity)
            raise

    async def _run(
        self, identity: str, reservation: Reservation, session: ConversationSession
    ) -> None:
        try:
            await self.release_local()
            await session.run()
        finally:
            try:
                await session.close()
            finally:
                self.sessions.pop(identity, None)
                self.lease.release(reservation.token)

    async def end(self, identity: str) -> None:
        reservation = self.sessions.get(identity)
        if reservation is None:
            return
        if reservation.expiry:
            reservation.expiry.cancel()
        if reservation.session:
            reservation.session.disconnected.set()
        if reservation.task:
            await asyncio.shield(reservation.task)
        else:
            try:
                if reservation.session:
                    await reservation.session.close()
            finally:
                self.sessions.pop(identity, None)
                self.lease.release(reservation.token)

    async def close(self) -> None:
        for identity in tuple(self.sessions):
            await self.end(identity)


def same_origin(request: Request) -> None:
    expected = f"{request.url.scheme}://{request.headers.get('host')}"
    if request.headers.get("origin") != expected:
        raise HTTPException(403, "Conversation requests must come from this site")


async def bounded_body(request: Request, limit: int) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise HTTPException(413, "Conversation request is too large")
    return bytes(body)


def install(app: FastAPI, conversations: Conversations) -> None:
    @app.post("/api/conversations", response_model=Created)
    async def create(request: Request) -> Created:
        same_origin(request)
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise HTTPException(
                409, "Conversation runtimes currently require Apple Silicon"
            )
        try:
            settings = REQUEST.validate_json(
                await bounded_body(request, 10 * 1024 * 1024)
            )
            return conversations.create(settings)
        except ValidationError as exc:
            raise HTTPException(
                422,
                exc.errors(
                    include_input=False, include_context=False, include_url=False
                ),
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/conversations/{identity}/offer", response_model=Answer)
    async def offer(identity: str, request: Request) -> Answer:
        same_origin(request)
        try:
            value = Offer.model_validate_json(await bounded_body(request, 110000))
            return await conversations.offer(identity, value)
        except (ValidationError, ValueError) as exc:
            raise HTTPException(422, "Invalid WebRTC offer") from exc

    @app.post("/api/conversations/{identity}/interrupt")
    async def interrupt(identity: str, request: Request) -> None:
        same_origin(request)
        session = conversations.get(identity).session
        if session:
            await session.conversation.controller.interrupt()

    @app.patch("/api/conversations/{identity}/offer", status_code=204)
    async def update_ice(identity: str, request: Request) -> None:
        same_origin(request)
        session = conversations.get(identity).session
        if session is None:
            raise HTTPException(409, "Offer required before ICE updates")
        try:
            update = IceUpdate.model_validate_json(await bounded_body(request, 110000))
            await session.update_ice(update)
        except (ValidationError, ValueError, AssertionError, IndexError) as exc:
            raise HTTPException(422, "Invalid ICE update") from exc

    @app.delete("/api/conversations/{identity}", status_code=204)
    async def end(identity: str, request: Request) -> None:
        same_origin(request)
        await conversations.end(identity)
