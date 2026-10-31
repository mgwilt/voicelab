"""Shared evidence and residency behavior across synthesis implementations."""

import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from voicelab.synthesis.experiments import Audio, ModelEvidence, run_trial
from voicelab_core.synthesis import Completion, SynthesisMetrics
from voicelab_web.models.synthesis.service import Delivery, SynthesisService


class Request(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    text: str


class Backend:
    implementation = "test"
    precision = "float32"
    device = "cpu"

    def __init__(
        self, completion: Completion = "complete", *, fail: bool = False
    ) -> None:
        self.completion = completion
        self.fail = fail
        self.closed = False

    def load(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def synthesize(
        self, request: Request, emit: Callable[[Audio], None]
    ) -> SynthesisMetrics:
        emit(np.zeros(240, dtype=np.float32))
        if self.fail:
            raise RuntimeError("Deliberate generation failure")
        return SynthesisMetrics(
            0.01, 0.01, 1, 0.01, None, None, None, (), self.completion
        )


class SynthesisIntegrationTests(unittest.TestCase):
    def test_unverified_and_incomplete_evidence_are_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = ModelEvidence("test", "revision", "test", "revision", 24000, ())
            for completion in ("complete", "unverified", "incomplete"):
                output = root / completion
                backend = Backend(completion)
                if completion == "incomplete":
                    with self.assertRaisesRegex(RuntimeError, "partial audio retained"):
                        run_trial(backend, Request(text="Hi"), model, output, root)
                else:
                    run_trial(backend, Request(text="Hi"), model, output, root)
                report = json.loads((output / "run.json").read_text())
                self.assertEqual(report["status"], completion)
                self.assertTrue((output / "speech.wav").is_file())
                self.assertIn("output_sha256", report)
                with self.assertRaises(FileExistsError):
                    run_trial(backend, Request(text="Hi"), model, output, root)

    def test_failure_retains_partial_audio_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = ModelEvidence("test", "revision", "test", "revision", 24000, ())
            with self.assertRaisesRegex(RuntimeError, "Deliberate"):
                run_trial(
                    Backend(fail=True), Request(text="Hi"), model, root / "run", root
                )
            report = json.loads((root / "run/run.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertGreater((root / "run/speech.wav").stat().st_size, 44)

    def test_unload_closes_resident_backend_and_next_run_reloads(self) -> None:
        instances: list[Backend] = []

        def factory() -> Backend:
            instance = Backend()
            instances.append(instance)
            return instance

        service = SynthesisService(Request, factory)
        service.generate(Request(text="Hi"), Delivery())
        service.unload()
        self.assertTrue(instances[0].closed)
        self.assertIsNone(service.runtime)
        service.generate(Request(text="Again"), Delivery())
        self.assertEqual(len(instances), 2)
        self.assertFalse(instances[1].closed)
        service.unload()
