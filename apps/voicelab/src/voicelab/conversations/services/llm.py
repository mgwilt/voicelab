"""Local Qwen service consuming Pipecat context and emitting response frames."""

from collections.abc import Callable
from time import perf_counter
from typing import Literal, override

from pipecat.adapters.llm.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContextMessage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pydantic import BaseModel, ConfigDict, TypeAdapter

from voicelab.conversations.contracts import Notice
from voicelab.conversations.services.frames import ResponseTextFrame
from voicelab.conversations.services.state import CascadeState
from voicelab_inference.qwen_text.mlx.runtime import Message
from voicelab_models.qwen_text.model import CHECKPOINT_ID, TEMPERATURE, TOP_K, TOP_P


class TextMessage(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")
    role: Literal["system", "user", "assistant"]
    content: str


MESSAGES = TypeAdapter(list[TextMessage])


class QwenLLMService(LLMService[OpenAILLMAdapter]):
    def __init__(self, state: CascadeState) -> None:
        super().__init__(
            settings=LLMSettings(
                model=CHECKPOINT_ID,
                system_instruction=state.system.content,
                max_tokens=256,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                filter_incomplete_user_turns=None,
                user_turn_completion_config=None,
                use_provider_thinking_default=False,
            )
        )
        self.state = state

    @override
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        messages = MESSAGES.validate_python(frame.context.get_messages())
        if not messages or messages[-1].role != "user":
            return
        state, turn = self.state, self.state.turn
        user = Message("user", messages[-1].content)
        context = (state.system, *state.history.snapshot(), user)
        bounded: list[LLMContextMessage] = []
        for message in context:
            match message.role:
                case "system":
                    bounded.append({"role": "system", "content": message.content})
                case "user":
                    bounded.append({"role": "user", "content": message.content})
                case "assistant":
                    bounded.append({"role": "assistant", "content": message.content})
        frame.context.set_messages(bounded)
        state.user = user
        state.answer = ""
        state.generating = True
        state.response_started = perf_counter()
        await state.notice(
            Notice(type="text", speaker="user", text=user.content, final=True), turn
        )
        await state.notice(Notice(type="state", text="Thinking"), turn)
        await self.push_frame(LLMFullResponseStartFrame())
        first = True

        def generate(emit: Callable[[str], None]) -> None:
            for text in state.llm.generate(context, turn.cancelled.is_set):
                emit(text)

        try:
            async for text in state.stream(generate, turn):
                if first:
                    await state.notice(
                        Notice(
                            type="metrics",
                            stage="llm_first_token",
                            seconds=perf_counter() - state.response_started,
                        ),
                        turn,
                    )
                    first = False
                state.answer += text
                await state.notice(Notice(type="text", text=text), turn)
            if state.current(turn) and state.answer.strip():
                await state.notice(
                    Notice(
                        type="metrics",
                        stage="llm",
                        seconds=perf_counter() - state.response_started,
                    ),
                    turn,
                )
                await state.notice(
                    Notice(type="text", text=state.answer, final=True), turn
                )
                # One frame preserves the selected voice's whole-response semantics.
                await self.push_frame(ResponseTextFrame(state.answer, turn=turn))
                await self.push_frame(LLMFullResponseEndFrame())
        except InterruptedError:
            pass
        except Exception as exc:
            await self.push_error(str(exc), fatal=True)
        finally:
            if state.current(turn):
                state.generating = False
