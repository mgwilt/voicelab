"""Exercise phonon request, delivery, and cleanup without model weights."""

import unittest
from collections.abc import Generator

import numpy as np
from pydantic import ValidationError

from voicelab_inference.phonon.torch.completion import reached_eos
from voicelab_inference.phonon.torch.contracts import Audio, Chunk
from voicelab_inference.phonon.torch.runtime import PhononTorch
from voicelab_models.phonon.model import SAMPLE_RATE, Request


class FakeBackend:
    def __init__(self, *, invalid: bool = False) -> None:
        self.invalid = invalid
        self.generator_closed = False
        self.closed = False

    def generate(self, request: Request) -> Generator[Chunk]:
        del request
        try:
            audio = np.full(24, np.nan if self.invalid else 0.25, dtype=np.float32)
            yield Chunk(audio, frames=1, eos=False)
            yield Chunk(audio, frames=1, eos=True)
        finally:
            self.generator_closed = True

    def close(self) -> None:
        self.closed = True


class AdapterTests(unittest.TestCase):
    def test_request_rejects_blank_and_unknown_fields(self) -> None:
        for value in ({"text": " "}, {"text": "hello", "unsupported": 1}):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                Request.model_validate(value)

    def test_metrics_use_actual_emitted_samples(self) -> None:
        backend = FakeBackend()
        runtime = PhononTorch(factory=lambda: backend)
        chunks: list[Audio] = []
        result = runtime.synthesize(Request(text="hello"), chunks.append)
        self.assertEqual(result.audio_seconds, 48 / SAMPLE_RATE)
        self.assertTrue(result.eos)
        self.assertEqual(len(result.arrivals), 2)
        self.assertTrue(backend.generator_closed)
        runtime.close()
        self.assertTrue(backend.closed)

    def test_cancel_closes_generator_and_allows_next_request(self) -> None:
        backend = FakeBackend()
        runtime = PhononTorch(factory=lambda: backend)

        def cancel(_audio: Audio) -> None:
            raise InterruptedError("cancelled")

        with self.assertRaises(InterruptedError):
            runtime.synthesize(Request(text="hello"), cancel)
        self.assertTrue(backend.generator_closed)
        result = runtime.synthesize(Request(text="try again"), lambda _audio: None)
        self.assertTrue(result.eos)
        runtime.close()

    def test_nonfinite_audio_is_not_delivered(self) -> None:
        backend = FakeBackend(invalid=True)
        runtime = PhononTorch(factory=lambda: backend)
        chunks: list[Audio] = []
        with self.assertRaises(ValueError):
            runtime.synthesize(Request(text="hello"), chunks.append)
        self.assertEqual(chunks, [])
        self.assertTrue(backend.generator_closed)
        runtime.close()


class CompletionTests(unittest.TestCase):
    def test_tail_must_finish_even_if_threshold_was_crossed(self) -> None:
        self.assertFalse(reached_eos((0.0, 5.0), 1))
        self.assertTrue(reached_eos((0.0, 5.0, 0.0), 1))
        self.assertFalse(reached_eos((0.0, 4.0, 0.0), 1))
