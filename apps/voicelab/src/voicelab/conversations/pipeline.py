"""Transport-independent Pipecat composition and model ownership."""

import asyncio
from contextlib import suppress
from typing import Protocol

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIProcessor
from pipecat.workers.runner import WorkerRunner

from voicelab.conversations.contracts import CascadeConfig, ConversationSpec, Notify
from voicelab.conversations.processor import ConversationProcessor
from voicelab.conversations.services.controller import CascadeController
from voicelab.conversations.services.llm import QwenLLMService
from voicelab.conversations.services.state import CascadeState
from voicelab.conversations.services.stt import WhisperSTTService
from voicelab.conversations.services.tts import LocalTTSService


class Controller(Protocol):
    ready: bool
    loaded: asyncio.Event

    async def interrupt(self) -> None: ...
    async def close(self) -> None: ...


class ConversationPipeline:
    def __init__(
        self,
        spec: ConversationSpec,
        input_processor: FrameProcessor,
        output_processor: FrameProcessor,
        notify: Notify,
        *,
        rtvi: RTVIProcessor | None = None,
        cascade_state: CascadeState | None = None,
    ) -> None:
        self.controller: Controller
        processors: list[FrameProcessor]
        if isinstance(spec, CascadeConfig):
            if cascade_state is None:
                from voicelab_inference.qwen_text.mlx.runtime import QwenTextMLX
                from voicelab_inference.whisper.mlx.runtime import WhisperMLX

                cascade_state = CascadeState(
                    WhisperMLX(), QwenTextMLX(), spec.voice, spec.system_prompt, notify
                )
            controller = CascadeController(cascade_state)
            self.controller = controller
            # The user aggregator owns Silero and Smart Turn; VAD frames travel
            # upstream to the segmented STT service without blocking input audio.
            user = LLMUserAggregator(
                LLMContext(),
                params=LLMUserAggregatorParams(
                    vad_analyzer=SileroVADAnalyzer(sample_rate=16000),
                    empty_user_turn=None,
                ),
            )
            processors = [
                input_processor,
                controller,
                WhisperSTTService(cascade_state),
                user,
                QwenLLMService(cascade_state),
                LocalTTSService(cascade_state),
                output_processor,
            ]
            rate = 16000
        else:
            native = ConversationProcessor(spec, notify)
            self.controller = native
            processors = [input_processor, native, output_processor]
            rate = spec.input_rate
        self.graph = Pipeline(processors)
        self.started = False
        self.close_task: asyncio.Task[None] | None = None
        self.worker = PipelineWorker(
            self.graph,
            params=PipelineParams(
                audio_in_sample_rate=rate, audio_out_sample_rate=24000
            ),
            enable_rtvi=rtvi is not None,
            rtvi_processor=rtvi,
            idle_timeout_secs=None,
        )
        self.runner = WorkerRunner(handle_sigint=False)

    async def run(self, disconnected: asyncio.Event) -> None:
        if self.started or self.close_task is not None:
            raise RuntimeError("A conversation pipeline can only run once")
        self.started = True
        await self.runner.add_workers(self.worker)
        running = asyncio.create_task(self.runner.run())
        ended = asyncio.create_task(disconnected.wait())
        loaded = asyncio.create_task(self.controller.loaded.wait())
        try:
            done, _ = await asyncio.wait(
                (running, ended, loaded),
                timeout=1800,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if loaded in done and not running.done() and not ended.done():
                await asyncio.wait(
                    (running, ended),
                    timeout=120,
                    return_when=asyncio.FIRST_COMPLETED,
                )
        finally:
            ended.cancel()
            loaded.cancel()
            await asyncio.gather(ended, loaded, return_exceptions=True)
            await self.worker.cancel()
            with suppress(asyncio.CancelledError):
                await running
            await self.close()

    async def close(self) -> None:
        if self.close_task is None:
            self.close_task = asyncio.create_task(self._close())
        await asyncio.shield(self.close_task)

    async def _close(self) -> None:
        try:
            await self.controller.close()
        finally:
            # Failed signaling never starts a worker, so it cannot clean up the graph.
            if not self.started:
                await self.graph.cleanup()
