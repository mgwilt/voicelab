"""Explicit construction; model adapters do not depend on one another."""

from collections.abc import Callable, Mapping

from pydantic import BaseModel

from voicelab.conversations.contracts import CascadeConfig, ConversationSpec, Voice
from voicelab_web.catalog_types import ModelId
from voicelab_web.conversations.requests import ConversationRequest


def create_backend(
    request: ConversationRequest,
    voices: Mapping[ModelId, Callable[[BaseModel], Voice]],
) -> ConversationSpec:
    match request.model:
        case "nemotron":
            from voicelab.conversations.adapters.nemotron import NemotronDuplex

            return NemotronDuplex(request.settings.prompt, request.settings.seed)
        case "personaplex":
            from voicelab.conversations.adapters.personaplex import (
                PersonaPlexDuplex,
            )

            return PersonaPlexDuplex(request.settings)
        case "minicpm":
            from voicelab.conversations.adapters.minicpm import MiniCPMConversation

            return MiniCPMConversation(
                request.settings, duplex=request.mode == "duplex"
            )
        case "qwen_omni":
            from voicelab.conversations.adapters.qwen_omni import QwenTurns

            return QwenTurns(request.settings)
        case "ming_audio":
            from voicelab.conversations.adapters.ming_audio import MingTurns

            return MingTurns(request.settings)
        case _:
            return CascadeConfig(
                voices[request.model](request.settings), request.system_prompt
            )
