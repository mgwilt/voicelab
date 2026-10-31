"""Explicit construction; model adapters do not depend on one another."""

from collections.abc import Callable, Mapping

from pydantic import BaseModel

from voicelab_web.catalog_types import ModelId
from voicelab_web.conversations.adapters.synthesis import Voice
from voicelab_web.conversations.contracts import Backend
from voicelab_web.conversations.requests import ConversationRequest


def create_backend(
    request: ConversationRequest,
    voices: Mapping[ModelId, Callable[[BaseModel], Voice]],
) -> Backend:
    match request.model:
        case "nemotron":
            from voicelab_web.conversations.adapters.nemotron import NemotronDuplex

            return NemotronDuplex(request.settings)
        case "personaplex":
            from voicelab_web.conversations.adapters.personaplex import (
                PersonaPlexDuplex,
            )

            return PersonaPlexDuplex(request.settings)
        case "minicpm":
            from voicelab_web.conversations.adapters.minicpm import MiniCPMConversation

            return MiniCPMConversation(
                request.settings, duplex=request.mode == "duplex"
            )
        case "qwen_omni":
            from voicelab_web.conversations.adapters.qwen_omni import QwenTurns

            return QwenTurns(request.settings)
        case "ming_audio":
            from voicelab_web.conversations.adapters.ming_audio import MingTurns

            return MingTurns(request.settings)
        case _:
            from voicelab_web.conversations.adapters.cascade import Cascade

            return Cascade(
                voices[request.model](request.settings), request.system_prompt
            )
