"""One peer connection and pipeline; ownership ends after accelerator cleanup."""

import asyncio

from aiortc.sdp import candidate_from_sdp
from pipecat.processors.frameworks.rtvi import RTVIProcessor
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from voicelab.conversations.contracts import CascadeConfig, ConversationSpec, Notice
from voicelab.conversations.pipeline import ConversationPipeline
from voicelab_web.conversations.transport.schemas import Answer, IceUpdate, Offer


class ConversationSession:
    def __init__(self, backend: ConversationSpec) -> None:
        self.connection = SmallWebRTCConnection(
            ice_servers=[], connection_timeout_secs=30
        )
        self.rtvi = RTVIProcessor()
        rate = 16000 if isinstance(backend, CascadeConfig) else backend.input_rate

        async def notify(event: Notice) -> None:
            await self.rtvi.send_server_message(event.model_dump())

        transport = SmallWebRTCTransport(
            self.connection,
            TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_stream_on_start=False,
                audio_in_sample_rate=rate,
                audio_out_sample_rate=24000,
            ),
        )
        self.conversation = ConversationPipeline(
            backend, transport.input(), transport.output(), notify, rtvi=self.rtvi
        )
        self.offered = False
        self.close_task: asyncio.Task[None] | None = None
        self.disconnected = asyncio.Event()

        async def disconnected(
            _transport: SmallWebRTCTransport, _connection: SmallWebRTCConnection
        ) -> None:
            self.disconnected.set()

        async def ready(_rtvi: RTVIProcessor) -> None:
            await self.rtvi.send_server_message(
                Notice(
                    type="state",
                    text="Listening"
                    if self.conversation.controller.ready
                    else "Loading models",
                ).model_dump()
            )

        transport.add_event_handler("on_client_disconnected", disconnected)
        self.rtvi.add_event_handler("on_client_ready", ready)

    async def offer(self, offer: Offer) -> Answer:
        if self.offered or offer.pc_id is not None or offer.restart_pc:
            raise ValueError("Start a new conversation to reconnect")
        self.offered = True
        await self.connection.initialize(offer.sdp, offer.type)
        return Answer.model_validate(self.connection.get_answer())

    async def run(self) -> None:
        try:
            await self.conversation.run(self.disconnected)
        finally:
            await self.close()

    async def update_ice(self, update: IceUpdate) -> None:
        if update.pc_id != self.connection.pc_id:
            raise ValueError("ICE update belongs to another peer")
        for item in update.candidates:
            candidate = candidate_from_sdp(item.candidate) if item.candidate else None
            if candidate is not None:
                candidate.sdpMid = item.sdp_mid
                candidate.sdpMLineIndex = item.sdp_mline_index
            await self.connection.add_ice_candidate(candidate)

    async def close(self) -> None:
        if self.close_task is None:
            self.close_task = asyncio.create_task(self._close())
        await asyncio.shield(self.close_task)

    async def _close(self) -> None:
        try:
            await self.conversation.close()
        finally:
            await self.connection.disconnect()
