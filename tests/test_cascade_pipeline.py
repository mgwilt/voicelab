"""Headless conversation services preserve synthesis, history, and cancellation."""

import asyncio
import threading
import unittest
from collections.abc import Iterator
from typing import override

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    LLMContextFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voicelab.conversations.contracts import (
    Audio,
    Cancelled,
    CascadeConfig,
    Emit,
    Notice,
)
from voicelab.conversations.pipeline import ConversationPipeline
from voicelab.conversations.services.state import CascadeState
from voicelab_core.synthesis import SynthesisMetrics
from voicelab_inference.qwen_text.mlx.runtime import Message


class Transcriber:
    closed = False

    def load(self) -> None:
        pass

    def transcribe(self, pcm: bytes) -> str:
        return "Hello"

    def close(self) -> None:
        self.closed = True


class Generator:
    closed = False

    def __init__(self) -> None:
        self.contexts: list[tuple[Message, ...]] = []

    def load(self) -> None:
        pass

    def generate(
        self, messages: tuple[Message, ...], cancelled: Cancelled
    ) -> Iterator[str]:
        self.contexts.append(messages)
        yield "A complete "
        yield "answer."

    def close(self) -> None:
        self.closed = True


class Voice:
    closed = False
    sample_rate = 24000

    def __init__(self, hold: bool = False, sample_rate: int = 24000) -> None:
        self.texts: list[str] = []
        self.hold = hold
        self.sample_rate = sample_rate
        self.release = threading.Event()

    def speak(self, text: str, emit: Emit, cancelled: Cancelled) -> SynthesisMetrics:
        self.texts.append(text)
        emit(Audio(bytes(4800), self.sample_rate))
        if self.hold:
            while not self.release.wait(0.01):
                if cancelled():
                    raise InterruptedError("cancelled")
        if cancelled():
            raise InterruptedError("cancelled")
        return SynthesisMetrics(
            generation_seconds=0.1,
            audio_seconds=0.1,
            output_rtf=1.0,
            first_pcm_seconds=0.01,
            codec_seconds=None,
            frames=1,
            eos=True,
            arrivals=(),
        )

    def close(self) -> None:
        self.closed = True


class PassThrough(FrameProcessor):
    @override
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class Speaker(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.audio = asyncio.Event()
        self.drained = asyncio.Event()
        self.responses = 0
        self.sample_rates: set[int] = set()

    @override
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            self.audio.set()
            self.sample_rates.add(frame.sample_rate)
        elif isinstance(frame, TTSStoppedFrame):
            self.responses += 1
            await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
            self.drained.set()
        await self.push_frame(frame, direction)


async def notify(event: Notice) -> None:
    pass


class CascadeTests(unittest.IsolatedAsyncioTestCase):
    async def test_headless_services_keep_whole_responses_and_bounded_history(
        self,
    ) -> None:
        stt, llm, voice = Transcriber(), Generator(), Voice(sample_rate=44100)
        state = CascadeState(stt, llm, voice, "Be brief.", notify)
        output = Speaker()
        pipeline = ConversationPipeline(
            CascadeConfig(voice, "Be brief."),
            PassThrough(),
            output,
            notify,
            cascade_state=state,
        )
        ended = asyncio.Event()
        running = asyncio.create_task(pipeline.run(ended))
        try:
            await asyncio.wait_for(pipeline.controller.loaded.wait(), 5)
            for index in range(8):
                output.drained.clear()
                await pipeline.worker.queue_frame(
                    LLMContextFrame(
                        LLMContext([{"role": "user", "content": f"Question {index}"}])
                    )
                )
                await asyncio.wait_for(output.drained.wait(), 5)
                # The acknowledgement flows asynchronously back through the pipeline.
                for _ in range(100):
                    if len(state.history.snapshot()) == min((index + 1) * 2, 12):
                        break
                    await asyncio.sleep(0.001)
            self.assertEqual(voice.texts, ["A complete answer."] * 8)
            self.assertEqual(len(state.history.snapshot()), 12)
            self.assertEqual(len(llm.contexts[-1]), 14)
            self.assertEqual(output.responses, 8)
            self.assertEqual(output.sample_rates, {44100})
        finally:
            ended.set()
            await asyncio.wait_for(running, 5)
        self.assertTrue(stt.closed and llm.closed and voice.closed)

    async def test_interrupt_discards_unplayed_exchange_and_next_turn_succeeds(
        self,
    ) -> None:
        voice = Voice(hold=True)
        state = CascadeState(Transcriber(), Generator(), voice, "Be brief.", notify)
        output = Speaker()
        pipeline = ConversationPipeline(
            CascadeConfig(voice, "Be brief."),
            PassThrough(),
            output,
            notify,
            cascade_state=state,
        )
        ended = asyncio.Event()
        running = asyncio.create_task(pipeline.run(ended))
        try:
            await asyncio.wait_for(pipeline.controller.loaded.wait(), 5)
            await pipeline.worker.queue_frame(
                LLMContextFrame(
                    LLMContext([{"role": "user", "content": "Interrupted question"}])
                )
            )
            await asyncio.wait_for(output.audio.wait(), 5)
            await pipeline.controller.interrupt()
            voice.release.set()
            self.assertEqual(state.history.snapshot(), ())
            await pipeline.worker.queue_frame(
                LLMContextFrame(
                    LLMContext([{"role": "user", "content": "Next question"}])
                )
            )
            await asyncio.wait_for(output.drained.wait(), 5)
            for _ in range(100):
                if state.history.snapshot():
                    break
                await asyncio.sleep(0.001)
            self.assertEqual(state.history.snapshot()[0].content, "Next question")
            self.assertEqual(len(state.history.snapshot()), 2)
        finally:
            ended.set()
            voice.release.set()
            await asyncio.wait_for(running, 5)

    async def test_close_before_signaling_releases_models_and_turn_detectors(
        self,
    ) -> None:
        stt, llm, voice = Transcriber(), Generator(), Voice()
        state = CascadeState(stt, llm, voice, "Be brief.", notify)
        pipeline = ConversationPipeline(
            CascadeConfig(voice, "Be brief."),
            PassThrough(),
            Speaker(),
            notify,
            cascade_state=state,
        )
        await asyncio.gather(pipeline.close(), pipeline.close())
        self.assertTrue(stt.closed and llm.closed and voice.closed)
        with self.assertRaisesRegex(RuntimeError, "only run once"):
            await pipeline.run(asyncio.Event())
