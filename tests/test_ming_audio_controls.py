"""Validate generation controls before a local model can load."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pydantic import ValidationError

from voicelab.models.ming_audio.runs import Report, execute_run
from voicelab_inference.ming_audio.mlx.contracts import Response
from voicelab_models.ming_audio.model import Request


class MingAudioControlTests(unittest.TestCase):
    def test_defaults_keep_existing_sampler_settings(self) -> None:
        request = Request(task="tts", prompt="Hello.")
        self.assertEqual(request.seed, 0)
        self.assertEqual(request.max_decode_steps, 375)
        self.assertEqual(request.cfg_scale, 2.0)
        self.assertEqual(request.flow_steps, 10)
        self.assertEqual(request.sigma, 0.25)
        self.assertEqual(request.temperature, 0.0)

    def test_ranges_and_types_are_strict(self) -> None:
        for field, value in (
            ("seed", -1),
            ("seed", 2**32),
            ("seed", 1.5),
            ("max_decode_steps", 0),
            ("max_decode_steps", 1501),
            ("flow_steps", 0),
            ("flow_steps", 101),
            ("cfg_scale", -0.1),
            ("sigma", float("inf")),
            ("temperature", float("nan")),
        ):
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(ValidationError),
            ):
                Request.model_validate(
                    {"task": "tts", "prompt": "Hello.", field: value}
                )

    def test_controls_round_trip_in_report_input(self) -> None:
        request = Request(
            task="tts",
            prompt="Hello.",
            seed=42,
            max_decode_steps=100,
            cfg_scale=0.0,
            flow_steps=5,
            sigma=0.5,
            temperature=1.0,
        )
        self.assertEqual(
            Request.model_validate_json(request.model_dump_json()), request
        )

    def test_failed_run_retains_sampler_evidence_and_closes(self) -> None:
        class FailingRuntime:
            load_seconds = 0.1

            def __init__(self) -> None:
                self.closed = False

            def load(self) -> None:
                pass

            def respond(self, request: Request) -> Response:
                del request
                raise RuntimeError("sampling failed")

            def close(self) -> None:
                self.closed = True

        runtime = FailingRuntime()
        request = Request(
            task="tts",
            prompt="Hello.",
            seed=53,
            max_decode_steps=70,
            cfg_scale=1.0,
            flow_steps=6,
            sigma=0.3,
            temperature=0.2,
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch(
                    "voicelab.models.ming_audio.runs.MingAudioMLX", return_value=runtime
                ),
                self.assertRaisesRegex(RuntimeError, "sampling failed"),
            ):
                execute_run(request, output)
            report = Report.model_validate_json((output / "run.json").read_bytes())
        self.assertTrue(runtime.closed)
        self.assertEqual(report.status, "failed")
        self.assertEqual(report.seed, request.seed)
        self.assertEqual(report.max_decode_steps, request.max_decode_steps)
        self.assertEqual(report.cfg_scale, request.cfg_scale)
        self.assertEqual(report.flow_steps, request.flow_steps)
        self.assertEqual(report.sigma, request.sigma)
        self.assertEqual(report.temperature, request.temperature)


if __name__ == "__main__":
    unittest.main()
