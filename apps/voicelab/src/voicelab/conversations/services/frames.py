"""Correlate completed local text with the turn that produced it."""

from dataclasses import dataclass

from pipecat.frames.frames import LLMTextFrame

from voicelab.conversations.services.state import Turn


@dataclass(kw_only=True)
class ResponseTextFrame(LLMTextFrame):
    turn: Turn
