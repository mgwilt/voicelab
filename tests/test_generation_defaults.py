"""CLI defaults agree with validated requests and preserve explicit overrides."""

import sys
import unittest
from unittest.mock import patch

from voicelab.models.audio8.cli import main as audio8
from voicelab.models.drama.cli import main as drama
from voicelab.models.minicpm.cli import main as minicpm
from voicelab_models.audio8.model import Request as Audio8Request
from voicelab_models.drama.model import Request as DramaRequest
from voicelab_models.minicpm.model import Request as MiniCPMRequest


class GenerationDefaultsTests(unittest.TestCase):
    def test_synthesis_cli_defaults_and_overrides_reach_runtime(self) -> None:
        for name, main, expected in (
            ("audio8", audio8, Audio8Request(text="Hello")),
            ("drama", drama, DramaRequest(text="Hello")),
        ):
            for override in (False, True):
                arguments = [name, "--text", "Hello", "--output", "unused"]
                if override:
                    arguments.extend(["--top-k", "81"])
                with (
                    self.subTest(model=name, override=override),
                    patch.object(sys, "argv", arguments),
                    patch(f"voicelab.models.{name}.cli.run") as run,
                ):
                    main()
                    run.assert_called_once()
                    request: object = run.call_args.args[2]
                    self.assertEqual(
                        request,
                        expected.model_copy(update={"top_k": 81})
                        if override
                        else expected,
                    )

    def test_minicpm_cli_keeps_voice_defaults_in_turn_mode(self) -> None:
        for override in (False, True):
            arguments = ["minicpm", "--text", "Hello", "--output", "unused"]
            if override:
                arguments.extend(["--top-k", "81"])
            with (
                self.subTest(override=override),
                patch.object(sys, "argv", arguments),
                patch("voicelab.models.minicpm.runs.execute_run") as run,
                patch("builtins.print"),
            ):
                minicpm()
                run.assert_called_once()
                request: object = run.call_args.args[1]
                expected = MiniCPMRequest(mode="turn_based", text="Hello")
                self.assertEqual(
                    request,
                    expected.model_copy(update={"top_k": 81}) if override else expected,
                )
