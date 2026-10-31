"""Retain auxiliary model identities and frontend versions in run reports."""

import json
import tempfile
import unittest
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import numpy as np

from voicelab.runs.synthesis.execution import (
    Audio,
    AuxiliaryCheckpoint,
    ModelEvidence,
    run_trial,
)
from voicelab_core.synthesis import SynthesisMetrics
from voicelab_models.phonon.model import Request


class Backend:
    implementation = "metadata-test"
    precision = "float32"
    device = "cpu"

    def __init__(self, *, fail: bool) -> None:
        self.fail = fail

    def load(self) -> None:
        pass

    def close(self) -> None:
        pass

    def synthesize(
        self, request: Request, emit: Callable[[Audio], None]
    ) -> SynthesisMetrics:
        del request
        emit(np.zeros(240, dtype=np.float32))
        if self.fail:
            raise RuntimeError("Generation failed")
        return SynthesisMetrics(0.01, 0.01, 1, 0.01, None, None, True, ())


class SynthesisProvenanceTests(unittest.TestCase):
    def test_reports_retain_codec_and_dependency_versions_on_success_and_failure(
        self,
    ) -> None:
        codec = AuxiliaryCheckpoint("codec", "example/decoder", "a" * 40, "codec.bin")
        model = ModelEvidence(
            "example/model",
            "b" * 40,
            "example/weights",
            "c" * 40,
            24000,
            ("numpy",),
            (codec,),
        )
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            for fail in (False, True):
                with self.subTest(fail=fail):
                    output = workspace / str(fail)
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "Generation failed"):
                            run_trial(
                                Backend(fail=fail),
                                Request(text="Hi"),
                                model,
                                output,
                                workspace,
                            )
                    else:
                        run_trial(
                            Backend(fail=fail),
                            Request(text="Hi"),
                            model,
                            output,
                            workspace,
                        )
                    report = json.loads((output / "run.json").read_text())
                    self.assertEqual(
                        report["auxiliary_checkpoints"],
                        [
                            {
                                "role": "codec",
                                "checkpoint": "example/decoder",
                                "revision": "a" * 40,
                                "filename": "codec.bin",
                            }
                        ],
                    )
                    self.assertEqual(
                        report["runtime"]["versions"]["numpy"], version("numpy")
                    )
                    self.assertEqual(report["status"], "failed" if fail else "complete")

    def test_models_without_auxiliary_checkpoints_keep_empty_default(self) -> None:
        model = ModelEvidence("model", "revision", "model", "revision", 24000, ())
        self.assertEqual(model.auxiliary_checkpoints, ())
