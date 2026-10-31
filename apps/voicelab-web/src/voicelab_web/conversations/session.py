"""One peer connection and pipeline; ownership ends after accelerator cleanup."""

import asyncio
from contextlib import suppress

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frameworks.rtvi import RTVIProcessor
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from voicelab_web.conversations.contracts import Backend, Notice
from voicelab_web.conversations.processor import ConversationProcessor
from voicelab_web.conversations.transport.schemas import Answer, Offer


class ConversationSession:
    def __init__(self, backend: Backend) -> None:
        self.connection = SmallWebRTCConnection(
            ice_servers=[], connection_timeout_secs=30
        )
        self.rtvi = RTVIProcessor()
        self.processor = ConversationProcessor(backend, self.rtvi)
        transport = SmallWebRTCTransport(
            self.connection,
            TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_stream_on_start=False,
                audio_in_sample_rate=backend.input_rate,
                audio_out_sample_rate=24000,
            ),
        )
        self.pipeline = PipelineWorker(
            Pipeline(
                [transport.input(), self.rtvi, self.processor, transport.output()]
            ),
            params=PipelineParams(
                audio_in_sample_rate=backend.input_rate, audio_out_sample_rate=24000
            ),
            enable_rtvi=False,
            idle_timeout_secs=None,
        )
        self.runner = WorkerRunner(handle_sigint=False)
        self.offered = False
        self.close_task: asyncio.Task[None] | None = None
        self.disconnected = asyncio.Event()

        async def disconnected(
            _transport: SmallWebRTCTransport, _connection: SmallWebRTCConnection
        ) -> None:
            self.disconnected.set()

        async def ready(_rtvi: RTVIProcessor) -> None:
            await self.rtvi.set_bot_ready()
            await self.rtvi.send_server_message(
                Notice(
                    type="state",
                    text="Listening" if self.processor.ready else "Loading models",
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
        await self.runner.add_workers(self.pipeline)
        running = asyncio.create_task(self.runner.run())
        disconnected = asyncio.create_task(self.disconnected.wait())
        loaded = asyncio.create_task(self.processor.loaded.wait())
        try:
            done, _ = await asyncio.wait(
                (running, disconnected, loaded),
                timeout=1800,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if loaded in done and not running.done() and not disconnected.done():
                await asyncio.wait(
                    (running, disconnected),
                    timeout=120,
                    return_when=asyncio.FIRST_COMPLETED,
                )
        finally:
            disconnected.cancel()
            loaded.cancel()
            await asyncio.gather(disconnected, loaded, return_exceptions=True)
            await self.pipeline.cancel()
            with suppress(asyncio.CancelledError):
                await running
            await self.close()

    async def close(self) -> None:
        if self.close_task is None:
            self.close_task = asyncio.create_task(self._close())
        await asyncio.shield(self.close_task)

    async def _close(self) -> None:
        try:
            await self.processor.close()
        finally:
            await self.connection.disconnect()
