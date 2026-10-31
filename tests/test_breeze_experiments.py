"""Portable request and immutable experiment evidence contracts."""

import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from voicelab.breeze.experiments import run_trial
from voicelab_core.synthesis import ChunkArrival, SynthesisMetrics
from voicelab_inference.breeze.mlx.audio import Audio
from voicelab_models.breeze.synthesis import SynthesisRequest


class FakeBackend:
    implementation = "fake"
    precision = "float32"
    device = "cpu"

    def __init__(self, *, eos: bool) -> None:
        self.eos = eos

    def synthesize(
        self, request: SynthesisRequest, emit: Callable[[Audio], None]
    ) -> SynthesisMetrics:
        emit(np.zeros(2400, dtype=np.float32))
        return SynthesisMetrics(
            0.05, 0.1, 0.5, 0.02, 0.01, 1, self.eos, (ChunkArrival(0.02, 2400),)
        )


class ExperimentTests(unittest.TestCase):
    def test_incomplete_preserves_audio_metrics_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            runtime = FakeBackend(eos=False)
            request = SynthesisRequest(text="Hello")
            with self.assertRaisesRegex(RuntimeError, "Frame limit"):
                run_trial(runtime, request, output, Path.cwd())
            report = json.loads((output / "run.json").read_text())
            self.assertEqual(report["status"], "incomplete")
            self.assertFalse(report["metrics"]["eos"])
            self.assertGreater((output / "speech.wav").stat().st_size, 44)
            with self.assertRaises(FileExistsError):
                run_trial(runtime, request, output, Path.cwd())

    def test_request_boundary(self) -> None:
        invalid = [
            '{"text":" "}',
            '{"text":"Hello","instruction":" "}',
            '{"text":"Hello","max_frames":0}',
            '{"text":"Hello","seed":-1}',
            '{"text":"Hello","cfg_scale":NaN}',
            '{"text":"Hello","speaker":0}',
            '{"text":"Hello","max_frames":"20"}',
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                SynthesisRequest.model_validate_json(value)
