"""Source-defined Nemotron tool insertion and safe supplied-response boundaries."""

import unittest

from pydantic import ValidationError

from voicelab_inference.nemotron.mlx.tool_calls import detect_call
from voicelab_inference.nemotron.mlx.tool_result import ToolResult
from voicelab_models.nemotron.tools import (
    Function,
    SuppliedResponse,
    Tool,
    ToolSettings,
    ToolTimeline,
)


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if text in {"<SPECIAL_20>", "<SPECIAL_21>"}:
            return [100 if text == "<SPECIAL_20>" else 101]
        return [ord(character) for character in text]

    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        return '{"name":"lookup","arguments":"{\\"query\\":\\"test\\"}"}'


class NemotronToolTests(unittest.TestCase):
    def test_only_response_tokens_are_forced(self) -> None:
        plan = ToolTimeline(call_start=7, call_tokens=3, response_tokens=(12, 13))
        self.assertEqual(plan.inserted_frames, 8)
        self.assertEqual(plan.response_start, 12)
        self.assertEqual(
            [plan.forced_token(i) for i in range(7, 16)],
            [None, None, None, None, None, 12, 13, None, None],
        )

    def test_complete_call_and_string_arguments_are_parsed(self) -> None:
        call = detect_call((0, 100, 1, 2, 101, 0), FakeTokenizer())
        self.assertEqual((call.start_frame, call.end_frame), (1, 4))
        self.assertEqual(call.call.arguments, {"query": "test"})
        with self.assertRaisesRegex(ValueError, "complete tool call"):
            detect_call((0, 100, 1), FakeTokenizer())

    def test_response_name_must_match_declared_tool(self) -> None:
        tool = Tool(
            function=Function(
                name="lookup",
                description="Look up a value",
                parameters={"type": "object"},
            )
        )
        with self.assertRaises(ValidationError):
            ToolSettings(
                tools=(tool,),
                api_response=SuppliedResponse(tool_name="other", response=1),
            )
        settings = ToolSettings(
            tools=(tool,),
            api_response=SuppliedResponse(tool_name="lookup", response={"value": 73}),
        )
        self.assertEqual(
            settings.response_text(), '<TOOL_RESPONSE>[{"value": 73}]</TOOL_RESPONSE>'
        )
        self.assertIn(
            '<AVAILABLE_TOOLS>[{"description":', settings.system_prompt("Help.")
        )
        with self.assertRaises(ValidationError):
            ToolSettings(tools=(tool, tool), api_response=settings.api_response)

    def test_nonfinite_and_nonascii_payloads_are_rejected(self) -> None:
        tool = Tool(function=Function(name="lookup", description="Read", parameters={}))
        for payload in (float("nan"), float("inf"), "café"):
            with self.assertRaises(ValidationError):
                ToolSettings(
                    tools=(tool,),
                    api_response=SuppliedResponse(tool_name="lookup", response=payload),
                )

    def test_missing_end_response_or_continuation_is_incomplete(self) -> None:
        call = detect_call((100, 1, 101), FakeTokenizer())
        for eotr, text, expected in (
            (False, "Some text", False),
            (True, "", False),
            (True, "The answer is 73", True),
        ):
            result = ToolResult(call, text, text, (), 10, 20, eotr, 1.0, 2.0, 3.0)
            self.assertEqual(result.completed, expected)

    def test_timeline_boundaries(self) -> None:
        for count in (0, 513):
            with self.assertRaises(ValueError):
                ToolTimeline(0, count, (1,))
        with self.assertRaises(ValueError):
            ToolTimeline(-1, 1, (1,))
        with self.assertRaises(ValueError):
            ToolTimeline(0, 1, ())
