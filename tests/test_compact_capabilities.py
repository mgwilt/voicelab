"""Validate model-specific conditioning choices before runtime allocation."""

import base64
import io
import unittest
import wave

import numpy as np
from pydantic import ValidationError

from voicelab_inference.ming_tts.mlx.preparation import trim_reference_tail
from voicelab_models.audio import ReferenceAudio
from voicelab_models.audio8.model import Request as Audio8Request
from voicelab_models.kokoro.model import Request as KokoroRequest
from voicelab_models.kokoro.voices import VOICES as KOKORO_VOICES
from voicelab_models.ming_tts.model import Request as MingRequest
from voicelab_models.ming_tts.voices import VOICES as MING_VOICES
from voicelab_models.phonon.model import Request as PhononRequest
from voicelab_models.phonon.voices import VOICES as PHONON_VOICES


def reference() -> ReferenceAudio:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(bytes(4800))
    return ReferenceAudio(wav_base64=base64.b64encode(output.getvalue()).decode())


class ConditioningTests(unittest.TestCase):
    def test_reference_trim_preserves_original_and_keeps_tail_margin(self) -> None:
        audio = np.zeros(16000, dtype=np.float32)
        audio[1600:8000] = 0.25
        trimmed = trim_reference_tail(audio, 16000)
        self.assertEqual(trimmed.size, 8800)
        self.assertEqual(audio.size, 16000)
        np.testing.assert_array_equal(trimmed, audio[:8800])
        with self.assertRaisesRegex(ValueError, "amplitude"):
            trim_reference_tail(np.zeros(16000, dtype=np.float32), 16000)

    def test_reference_and_transcript_are_an_indivisible_input(self) -> None:
        audio = reference()
        for schema in (Audio8Request, MingRequest):
            for fields in (
                {"ref_audio": audio},
                {"ref_text": "Hello"},
                {"ref_audio": audio, "ref_text": " "},
            ):
                with (
                    self.subTest(schema=schema, fields=fields),
                    self.assertRaises(ValidationError),
                ):
                    schema.model_validate({"text": "New speech", **fields})
            request = schema(text="New speech", ref_audio=audio, ref_text="Hello")
            self.assertEqual(request.ref_audio, audio)

    def test_ming_rejects_conflicting_conditioning(self) -> None:
        invalid = (
            {"instruction": "Whisper", "emotion": "高兴"},
            {"voice": "US_Emma", "style": "A deep voice"},
            {"voice": "US_Emma", "ref_audio": reference(), "ref_text": "Hi"},
            {"voice": "invented"},
            {"language": "unsupported"},
        )
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                MingRequest.model_validate({"text": "Hello", **fields})

    def test_official_voice_catalogs_validate(self) -> None:
        for voice in PHONON_VOICES:
            PhononRequest(text="Hello", voice=voice)
        for voice in KOKORO_VOICES:
            KokoroRequest(text="Hello", voice=voice)
        for voice in MING_VOICES:
            MingRequest(text="Hello", voice=voice)

    def test_kokoro_blends_preserve_language(self) -> None:
        request = KokoroRequest(text="Hello", voice="af_heart,af_bella", lang_code="a")
        self.assertEqual(request.voice, "af_heart,af_bella")
        for voice, language in (("af_heart,jf_alpha", None), ("jf_alpha", "a")):
            with self.assertRaises(ValidationError):
                KokoroRequest.model_validate(
                    {"text": "Hello", "voice": voice, "lang_code": language}
                )

    def test_audio8_sampling_controls_reject_nonfinite_and_out_of_range(self) -> None:
        for fields in ({"top_p": float("nan")}, {"top_k": -1}, {"do_sample": "false"}):
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                Audio8Request.model_validate({"text": "Hello", **fields})
