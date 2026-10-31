"""Bounded, supplied tool responses for Nemotron's offline two-pass workflow."""

import json
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

SOURCE_REVISION = "097dfe9e2f55baf653b83035868bdc89849f1b47"
MAX_TOOL_TOKENS = 512


class Function(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    description: str = Field(min_length=1, max_length=2000)
    parameters: dict[str, JsonValue]


class Tool(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    type: Literal["function"] = "function"
    function: Function


class SuppliedResponse(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    tool_name: str = Field(min_length=1, max_length=128)
    response: JsonValue


class ToolSettings(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    tools: tuple[Tool, ...] = Field(min_length=1, max_length=8)
    api_response: SuppliedResponse

    @model_validator(mode="after")
    def validate_tools(self) -> Self:
        names = [tool.function.name for tool in self.tools]
        if len(set(names)) != len(names):
            raise ValueError("Tool names must be unique.")
        if self.api_response.tool_name not in names:
            raise ValueError("The supplied response must name one of the tools.")
        serialized = json.dumps(
            {
                "tools": [tool.model_dump() for tool in self.tools],
                "api_response": self.api_response.model_dump(),
            },
            ensure_ascii=False,
            allow_nan=False,
        )
        if not serialized.isascii():
            raise ValueError("Nemotron tool definitions and responses must be ASCII.")
        if len(serialized) > 16000:
            raise ValueError("Tool definitions and response exceed 16000 characters.")
        return self

    def system_prompt(self, message: str) -> str:
        """Render NVIDIA's system-only template without arbitrary template execution."""
        tools = ", ".join(
            json.dumps(tool.function.model_dump(), sort_keys=True)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
            .replace("'", "\\u0027")
            for tool in self.tools
        )
        return (
            message
            + ("\n\n" if message else "")
            + "You can use the following tools to assist the user if required:"
            + f"\n<AVAILABLE_TOOLS>[{tools}]</AVAILABLE_TOOLS>\n\n"
            + "If you decide to call any tool(s), use the following format:\n"
            + '<TOOLCALL>[{"name": "tool_name1", "arguments": "tool_args1"}, '
            + '{"name": "tool_name2", "arguments": "tool_args2"}]</TOOLCALL>\n\n'
            + "The user will execute tool-calls and return responses from tool(s) "
            + "in this format:\n"
            + '<TOOL_RESPONSE>[{"tool_response1"}, {"tool_response2"}]'
            + "</TOOL_RESPONSE>\n\n"
            + "Based on the tool responses, you can call additional tools if needed, "
            + "correct tool calls if any errors are found, or just respond to the user."
        )

    def response_text(self) -> str:
        return (
            "<TOOL_RESPONSE>["
            + json.dumps(self.api_response.response)
            + "]</TOOL_RESPONSE>"
        )


class ToolCall(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    name: str = Field(min_length=1, max_length=128)
    arguments: JsonValue


@dataclass(frozen=True)
class DetectedCall:
    start_frame: int
    end_frame: int
    call: ToolCall
    serialized: str
    token_count: int


@dataclass(frozen=True)
class ToolTimeline:
    """NVIDIA inserts call, response and EOTR at the same original-audio anchor."""

    call_start: int
    call_tokens: int
    response_tokens: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.call_start < 0 or not 0 < self.call_tokens <= MAX_TOOL_TOKENS:
            raise ValueError("Invalid tool call timeline.")
        if not 0 < len(self.response_tokens) <= MAX_TOOL_TOKENS:
            raise ValueError("Tool response must contain 1..512 tokens.")

    @property
    def response_start(self) -> int:
        return self.call_start + self.call_tokens + 2

    @property
    def inserted_frames(self) -> int:
        return self.call_tokens + len(self.response_tokens) + 3

    def forced_token(self, frame: int) -> int | None:
        index = frame - self.response_start
        return (
            self.response_tokens[index]
            if 0 <= index < len(self.response_tokens)
            else None
        )
