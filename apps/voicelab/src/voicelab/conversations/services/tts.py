"""Selected local voice as a Pipecat TTS service with explicit audio contexts."""

from collections.abc import AsyncGenerator, Callable
from time import perf_counter
from typing import override

from pipecat.frames.frames import Frame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, TTSService

from voicelab.conversations.contracts import Audio, Notice
from voicelab.conversations.services.frames import ResponseTextFrame
from voicelab.conversations.services.state import CascadeState, Turn
from voicelab_core.synthesis import SynthesisMetrics
from voicelab_inference.qwen_text.mlx.runtime import Message


class LocalTTSService(TTSService):
    def __init__(self, state: CascadeState) -> None:
        super().__init__(
            text_aggregation_mode=TextAggregationMode.TOKEN,
            push_start_frame=True,
            sample_rate=state.voice.sample_rate,
            settings=TTSSettings(model=None, voice=None, language=None),
        )
        self.state = state
        self.response_turn: Turn | None = None

    @override
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, ResponseTextFrame):
            if not self.state.current(frame.turn):
                return
            self.response_turn = frame.turn
        await super().process_frame(frame, direction)

    @override
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None]:
        state, turn = self.state, self.response_turn
        if turn is None or not state.current(turn):
            return
        user = state.user
        await state.notice(Notice(type="state", text="Speaking"), turn)
        first = True
        state.synthesizing = True

        def speak(emit: Callable[[Audio | Notice | SynthesisMetrics], None]) -> None:
            result = state.voice.speak(text, emit, turn.cancelled.is_set)
            emit(result)

        try:
            async for event in state.stream(speak, turn):
                if isinstance(event, Audio):
                    if first:
                        first = False
                        await state.notice(
                            Notice(
                                type="metrics",
                                stage="turn_to_first_audio",
                                seconds=perf_counter() - state.response_started,
                            ),
                            turn,
                        )
                    state.output_pending = True
                    yield TTSAudioRawFrame(
                        event.pcm, event.sample_rate, 1, context_id=context_id
                    )
                elif isinstance(event, Notice):
                    await state.notice(event, turn)
                else:
                    await state.notice(
                        Notice(
                            type="metrics",
                            stage="tts",
                            text=event.completion_state,
                            seconds=event.generation_seconds,
                            output_rtf=event.output_rtf,
                        ),
                        turn,
                    )
                    if (
                        state.current(turn)
                        and event.completion_state != "incomplete"
                        and user is not None
                    ):
                        state.history.stage(user, Message("assistant", text))
            if state.current(turn):
                yield TTSStoppedFrame(context_id=context_id)
        except InterruptedError:
            pass
        except Exception as exc:
            await self.push_error(str(exc), fatal=True)
        finally:
            if state.current(turn):
                state.synthesizing = False
