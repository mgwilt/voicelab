"""Validate one complete predicted tool call before accepting a supplied response."""

import json
from typing import Protocol

from pydantic import JsonValue, TypeAdapter

from voicelab_models.nemotron.tools import MAX_TOOL_TOKENS, DetectedCall, ToolCall


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...
    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str: ...


def detect_call(tokens: tuple[int, ...], tokenizer: Tokenizer) -> DetectedCall:
    markers = [
        tokenizer.encode(marker, add_special_tokens=False)
        for marker in ("<SPECIAL_20>", "<SPECIAL_21>")
    ]
    if any(len(ids) != 1 for ids in markers):
        raise RuntimeError("Nemotron function boundary markers must be single tokens.")
    start_id, end_id = markers[0][0], markers[1][0]
    start: int | None = None
    for frame, token in enumerate(tokens):
        if token == start_id and start is None:
            start = frame
        elif token == end_id and start is not None:
            raw = tokenizer.decode(
                list(tokens[start + 1 : frame]), skip_special_tokens=False
            )
            clean = raw.replace("<TOOLCALL>", "").replace("</TOOLCALL>", "").strip()
            calls = TypeAdapter(list[ToolCall]).validate_json(
                clean if clean.startswith("[") else f"[{clean}]", strict=True
            )
            if len(calls) != 1:
                raise ValueError(
                    "This supplied-response experiment requires exactly one tool call."
                )
            call = calls[0]
            if isinstance(call.arguments, str):
                call = call.model_copy(
                    update={
                        "arguments": TypeAdapter(JsonValue).validate_json(
                            call.arguments
                        )
                    }
                )
            json.dumps(call.model_dump(), allow_nan=False)
            count = len(tokenizer.encode(clean, add_special_tokens=False))
            if not 0 < count <= MAX_TOOL_TOKENS:
                raise ValueError("Predicted tool call exceeds 512 tokens.")
            return DetectedCall(start, frame, call, clean, count)
    raise ValueError(
        "Nemotron did not emit a complete tool call in the input timeline."
    )
