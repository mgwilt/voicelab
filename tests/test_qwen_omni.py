"""Weight-free Qwen request, completion, timing and retained-evidence checks."""

import threading
import unittest
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from voicelab.qwen_omni.experiments import Report, run_experiment
from voicelab_inference.qwen_omni.mlx.contracts import Completion
from voicelab_inference.qwen_omni.mlx.runtime import QwenOmniMLX
from voicelab_models.qwen_omni.model import HistoryTurn, Request


class FakeBackend:
    def __init__(self) -> None:
        self.closed = False
        self.complete = True
        self.fail = False

    def generate(
        self,
        request: Request,
        emit_text: Callable[[str], None],
        emit_audio: Callable[[NDArray[np.float32]], None],
        should_cancel: Callable[[], bool],
    ) -> Completion:
        if should_cancel():
            raise InterruptedError
        emit_text("Hello")
        if request.return_audio:
            emit_audio(np.full(24000, 0.25, dtype=np.float32))
        if self.fail:
            raise RuntimeError("decode failed")
        return Completion(
            "Hello", True, self.complete if request.return_audio else None, 2, 13
        )

    def close(self) -> None:
        self.closed = True


class QwenOmniTests(unittest.TestCase):
    def test_request_rejects_invalid_controls_and_history(self) -> None:
        for update in (
            {"seed": True},
            {"temperature": float("nan")},
            {"chunk_frames": 0},
            {"speaker": "Unknown"},
            {"talker_max_tokens": 1501},
            {"prompt": " "},
            {"history": (HistoryTurn(role="user", content="hello"),)},
            {
                "history": (
                    HistoryTurn(role="assistant", content="hello"),
                    HistoryTurn(role="user", content="hi"),
                )
            },
        ):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                Request.model_validate(update)

    def test_timings_exclude_load_and_mark_first_pcm_before_callback(self) -> None:
        backend = FakeBackend()
        runtime = QwenOmniMLX(factory=lambda: backend)
        clock = [10.0]

        def now() -> float:
            return clock[0]

        def audio(pcm: bytes) -> None:
            self.assertEqual(len(pcm), 48000)
            clock[0] = 12.0

        with patch("voicelab_inference.qwen_omni.mlx.runtime.perf_counter", now):
            response = runtime.respond(Request(), emit_audio=audio)
        self.assertEqual(response.metrics.first_pcm_seconds, 0)
        self.assertEqual(response.metrics.output_seconds, 1)
        self.assertEqual(response.metrics.rtf, 2)
        self.assertTrue(response.metrics.completed)

    def test_text_only_has_no_speech_timing_and_caps_are_not_eos(self) -> None:
        backend = FakeBackend()
        runtime = QwenOmniMLX(factory=lambda: backend)
        response = runtime.respond(Request(return_audio=False))
        self.assertIsNone(response.metrics.rtf)
        self.assertIsNone(response.metrics.first_pcm_seconds)
        self.assertIsNone(response.metrics.talker_eos)
        backend.complete = False
        self.assertFalse(runtime.respond(Request()).metrics.completed)

    def test_cancel_after_audio_then_reuse(self) -> None:
        backend = FakeBackend()
        runtime = QwenOmniMLX(factory=lambda: backend)
        cancel = threading.Event()
        with self.assertRaises(InterruptedError):
            runtime.respond(
                Request(),
                emit_audio=lambda _: cancel.set(),
                should_cancel=cancel.is_set,
            )
        self.assertTrue(runtime.respond(Request()).metrics.completed)
        runtime.close()
        self.assertTrue(backend.closed)

    def test_failed_and_capped_runs_retain_partial_audio_and_evidence(self) -> None:
        for failed in (False, True):
            backend = FakeBackend()
            backend.fail = failed
            backend.complete = False
            with TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                with self.assertRaises(RuntimeError):
                    run_experiment(
                        Request(seed=72),
                        output,
                        QwenOmniMLX(factory=lambda value=backend: value),
                    )
                report = Report.model_validate_json((output / "run.json").read_bytes())
                self.assertEqual(report.status, "failed" if failed else "incomplete")
                self.assertEqual(report.text, "Hello")
                self.assertEqual(report.request["seed"], 72)
                self.assertGreater((output / "response.wav").stat().st_size, 44)
                self.assertIsNotNone(report.output_sha256)
            self.assertTrue(backend.closed)
