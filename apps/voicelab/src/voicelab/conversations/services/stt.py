"""Whisper as Pipecat's segmented local STT service."""

from collections import deque
from collections.abc import AsyncGenerator
from functools import partial
from time import perf_counter
from typing import override

from pipecat.frames.frames import Frame, TranscriptionFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

from voicelab.conversations.contracts import Notice
from voicelab.conversations.services.state import CascadeState, Turn
from voicelab_models.whisper.model import CHECKPOINT_ID


class WhisperSTTService(SegmentedSTTService):
    def __init__(self, state: CascadeState) -> None:
        super().__init__(
            sample_rate=16000, settings=STTSettings(model=CHECKPOINT_ID, language=None)
        )
        self.state = state
        self.segments: deque[Turn] = deque()

    @override
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            if len(self.segments) >= 2:
                await self.push_error(
                    "Transcription cannot keep up with speech", fatal=True
                )
                return
            self.segments.append(self.state.turn)
        await super().process_frame(frame, direction)

    @property
    @override
    def wants_wav_segments(self) -> bool:
        return False

    @override
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None]:
        turn = self.segments.popleft() if self.segments else self.state.turn
        if not self.state.current(turn):
            return
        start = perf_counter()
        await self.state.notice(Notice(type="state", text="Transcribing"), turn)
        text = await self.state.worker.call(partial(self.state.stt.transcribe, audio))
        if self.state.current(turn):
            await self.state.notice(
                Notice(type="metrics", stage="stt", seconds=perf_counter() - start),
                turn,
            )
            if text:
                yield TranscriptionFrame(text, "user", time_now_iso8601())
