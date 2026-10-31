"""Exercise the run boundary without downloading or loading a model."""

import json
import tempfile
import unittest
import wave
from pathlib import Path

from voicelab.models.nemotron.runs import execute_run
from voicelab_core.audio import inspect_input
from voicelab_core.contracts import GenerationSettings, SpeechOutput
from voicelab_models.nemotron.model import MLX_8BIT, Checkpoint


def write_wave(path: Path, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\0\0" * 1600)


class FakeBackend:
    def synchronize(self) -> None:
        pass

    def generate(
        self, source: Path, destination: Path, settings: GenerationSettings
    ) -> SpeechOutput:
        write_wave(destination, rate=22050)
        return SpeechOutput(destination, "Hello", "Hi", 22050)


def fail_loading() -> FakeBackend:
    raise RuntimeError("Device unavailable")


class RunTests(unittest.TestCase):
    def test_completed_report_preserves_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.wav"
            write_wave(source)
            (root / "uv.lock").write_text("fixture")
            report = execute_run(
                source,
                root / "run",
                MLX_8BIT,
                GenerationSettings(),
                FakeBackend,
                backend_name="test/fake",
                workspace=root,
            )
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "completed")
            self.assertEqual(data["checkpoint"]["revision"], MLX_8BIT.revision)
            self.assertEqual(data["backend"], "test/fake")
            self.assertEqual(len(data["input"]["sha256"]), 64)
            self.assertGreaterEqual(data["generation_seconds"], 0)
            with self.assertRaises(FileExistsError):
                execute_run(
                    source,
                    root / "run",
                    MLX_8BIT,
                    GenerationSettings(),
                    FakeBackend,
                    backend_name="test/fake",
                    workspace=root,
                )

    def test_backend_failure_leaves_failed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.wav"
            write_wave(source)
            (root / "uv.lock").write_text("fixture")
            with self.assertRaisesRegex(RuntimeError, "Device unavailable"):
                execute_run(
                    source,
                    root / "run",
                    MLX_8BIT,
                    GenerationSettings(),
                    fail_loading,
                    backend_name="test/fake",
                    workspace=root,
                )
            data = json.loads((root / "run/run.json").read_text())
            self.assertEqual(data["status"], "failed")
            self.assertNotIn("generation_seconds", data)

    def test_wrong_sample_rate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "wrong.wav"
            write_wave(source, rate=24000)
            with self.assertRaisesRegex(ValueError, "16000"):
                inspect_input(source)

    def test_mutable_checkpoint_revision_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Checkpoint("example/model", "main", "8bit")

    def test_invalid_silence_is_rejected(self) -> None:
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GenerationSettings(trailing_silence_seconds=value)
