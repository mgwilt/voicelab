"""Call readiness, interruption policy, and playback acknowledgements."""

import asyncio
from contextlib import suppress
from typing import override

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voicelab.conversations.contracts import Notice
from voicelab.conversations.services.state import CascadeState


class CascadeController(FrameProcessor):
    def __init__(self, state: CascadeState) -> None:
        super().__init__()
        self.state = state
        self.ready = False
        self.loaded = asyncio.Event()
        self.loading: asyncio.Task[None] | None = None
        self.speech_bytes: int | None = None

    @override
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self.loading = self.create_task(self._load())
            return
        if isinstance(frame, (EndFrame, CancelFrame)):
            self.state.stopping = True
            self.state.turn.cancelled.set()
        elif isinstance(frame, InterruptionFrame):
            active = (
                self.state.generating
                or self.state.synthesizing
                or self.state.output_pending
            )
            interrupted = self.state.turn.number
            self.state.interrupt()
            if active:
                await self.state.notify(Notice(type="interrupted", turn=interrupted))
        elif isinstance(frame, UserStartedSpeakingFrame):
            self.speech_bytes = 0
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.speech_bytes = None
        elif isinstance(frame, InputAudioRawFrame):
            if not self.ready:
                return
            if frame.sample_rate != 16000 or frame.num_channels != 1:
                await self.push_error("Invalid cascade PCM format", fatal=True)
                return
            if self.speech_bytes is not None:
                self.speech_bytes += len(frame.audio)
                if self.speech_bytes > 16000 * 2 * 30:
                    await self.push_error(
                        "A spoken turn exceeds 30 seconds", fatal=True
                    )
                    return
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.state.output_pending = False
            if not self.state.stopping:
                self.state.history.played()
                await self.state.notice(Notice(type="state", text="Listening"))
        elif isinstance(frame, ErrorFrame):
            await self.state.notice(Notice(type="error", text=frame.error))
        await self.push_frame(frame, direction)

    async def _load(self) -> None:
        try:
            await self.state.notice(Notice(type="state", text="Loading local Whisper"))
            await self.state.worker.call(self.state.stt.load)
            if not self.state.current(self.state.turn):
                return
            await self.state.notice(Notice(type="state", text="Loading local Qwen"))
            await self.state.worker.call(self.state.llm.load)
            if not self.state.current(self.state.turn):
                return
            self.ready = True
            self.loaded.set()
            await self.state.notice(Notice(type="state", text="Listening"))
        except Exception as exc:
            await self.state.notice(Notice(type="error", text=str(exc)))
            await self.push_error(str(exc), fatal=True)

    async def interrupt(self) -> None:
        if self.ready:
            # Use the same frame path as automatic user-turn interruptions.
            active = (
                self.state.generating
                or self.state.synthesizing
                or self.state.output_pending
            )
            number = self.state.turn.number
            self.state.interrupt()
            await self.broadcast_interruption()
            if active:
                await self.state.notify(Notice(type="interrupted", turn=number))

    async def close(self) -> None:
        self.ready = False
        self.state.stopping = True
        self.state.turn.cancelled.set()
        if self.loading:
            self.loading.cancel()
            with suppress(asyncio.CancelledError):
                await self.loading
        await self.state.close()

    @override
    async def cleanup(self) -> None:
        await self.close()
        await super().cleanup()
