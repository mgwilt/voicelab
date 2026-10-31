"""Reference boundaries reject invalid recordings and reports retain only provenance."""

import base64
import io
import json
import unittest
import wave
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ValidationError

from voicelab.synthesis.evidence import request_evidence
from voicelab_inference.audio import reference_file, reference_pcm
from voicelab_models.audio import ReferenceAudio
from voicelab_models.breeze.synthesis import SynthesisRequest
from voicelab_models.sesame.model import ContextSegment
from voicelab_models.sesame.model import SynthesisRequest as SesameRequest


def recording(rate: int = 16000, channels: int = 1, seconds: float = 0.1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(
            np.full(int(rate * seconds) * channels, 8192, dtype="<i2").tobytes()
        )
    return buffer.getvalue()


def reference(data: bytes) -> ReferenceAudio:
    return ReferenceAudio(wav_base64=base64.b64encode(data).decode("ascii"))


class NestedReference(BaseModel):
    references: list[ReferenceAudio]


class ReferenceAudioTests(unittest.TestCase):
    def test_context_preserves_roles_and_rejects_ambiguous_or_excessive_audio(
        self,
    ) -> None:
        source = reference(recording())
        turns = (
            ContextSegment(audio=source, text="Hello", speaker=0),
            ContextSegment(audio=source, text="Hi", speaker=1),
        )
        request = SesameRequest(text="How are you?", context=turns)
        self.assertEqual(tuple(turn.speaker for turn in request.context), (0, 1))
        with self.assertRaises(ValidationError):
            SesameRequest(
                text="Hello", context=turns, ref_audio=source, ref_text="Hello"
            )
        long = ContextSegment(
            audio=reference(recording(seconds=16)), text="Long", speaker=0
        )
        with self.assertRaises(ValidationError):
            SesameRequest(text="Hello", context=(long, long))

    def test_validation_prevents_invalid_or_unbounded_wave(self) -> None:
        for data in (
            b"not a WAV",
            recording(channels=2),
            recording(seconds=0),
            recording(seconds=30.1),
            recording()[:-2],
        ):
            with self.subTest(length=len(data)), self.assertRaises(ValidationError):
                reference(data)
        with self.assertRaises(ValidationError):
            ReferenceAudio(wav_base64="!!!!")

    def test_resampling_preserves_duration_and_amplitude(self) -> None:
        source = reference(recording())
        samples = reference_pcm(source, 24000)
        self.assertEqual(samples.shape, (2400,))
        self.assertEqual(samples.dtype, np.float32)
        self.assertAlmostEqual(float(np.mean(samples[20:-20])), 0.25, places=3)
        self.assertEqual(source.sample_rate, 16000)

    def test_temporary_reference_lifetime_includes_failure(self) -> None:
        source = reference(recording())
        retained: Path | None = None
        with (
            self.assertRaisesRegex(RuntimeError, "cancel"),
            reference_file(source) as path,
        ):
            self.assertIsNotNone(path)
            if path is None:
                self.fail("Missing reference file")
            retained = Path(path)
            self.assertEqual(retained.read_bytes(), source.wav_bytes())
            raise RuntimeError("cancel")
        self.assertIsNotNone(retained)
        if retained is not None:
            self.assertFalse(retained.exists())

    def test_nested_evidence_omits_recording_bytes(self) -> None:
        source = reference(recording())
        result = json.dumps(request_evidence(NestedReference(references=[source])))
        self.assertIn("sha256", result)
        self.assertNotIn("wav_base64", result)
        self.assertNotIn(source.wav_base64, result)

    def test_breeze_requires_paired_reference_and_allows_clone_without_instruction(
        self,
    ) -> None:
        source = reference(recording())
        with self.assertRaises(ValidationError):
            SynthesisRequest(text="Hello", ref_audio=source)
        request = SynthesisRequest(
            text="Hello", instruction=None, ref_audio=source, ref_text="A reference."
        )
        self.assertIsNone(request.instruction)
