"""Test local Ming response ownership without weights or accelerators."""

import base64
import io
import unittest
import wave
from collections.abc import Callable

import numpy as np

from voicelab_inference.ming_audio.mlx.contracts import Output
from voicelab_inference.ming_audio.mlx.runtime import MingAudioMLX
from voicelab_models.ming_audio.model import Request


def request() -> Request:
    encoded = io.BytesIO()
    with wave.open(encoded, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes(3200))
    return Request(audio_wav_base64=base64.b64encode(encoded.getvalue()).decode())


class FakeBackend:
    def __init__(self, *, invalid: bool = False) -> None:
        self.closed = False
        self.called = False
        self.invalid = invalid

    def respond(self, request: Request, should_cancel: Callable[[], bool]) -> Output:
        del request
        self.called = True
        if should_cancel():
            raise InterruptedError("cancelled")
        audio = np.full(441, np.nan if self.invalid else 0.25, dtype=np.float32)
        return Output("hello", audio, completed=False)

    def close(self) -> None:
        self.closed = True


class MingAudioLocalTests(unittest.TestCase):
    def test_response_preserves_incomplete_and_measures_local_audio(self) -> None:
        backend = FakeBackend()
        runtime = MingAudioMLX(factory=lambda: backend)
        response = runtime.respond(request())
        self.assertEqual(response.text, "hello")
        self.assertEqual(len(response.pcm), 882)
        self.assertEqual(response.metrics.input_seconds, 0.1)
        self.assertEqual(response.metrics.output_seconds, 0.01)
        self.assertFalse(response.metrics.completed)
        runtime.close()
        self.assertTrue(backend.closed)

    def test_cancel_does_not_load_model(self) -> None:
        loaded = False

        def factory() -> FakeBackend:
            nonlocal loaded
            loaded = True
            return FakeBackend()

        with self.assertRaises(InterruptedError):
            MingAudioMLX(factory=factory).respond(request(), lambda: True)
        self.assertFalse(loaded)

    def test_nonfinite_output_is_rejected(self) -> None:
        backend = FakeBackend(invalid=True)
        runtime = MingAudioMLX(factory=lambda: backend)
        with self.assertRaises(ValueError):
            runtime.respond(request())
        runtime.close()
        self.assertTrue(backend.closed)
