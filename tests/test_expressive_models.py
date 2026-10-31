"""Local expressive model request, completion, and resource invariants."""

import base64
import io
import json
import tempfile
import unittest
import wave
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from voicelab.models.drama.history import load_previous_takes
from voicelab_inference.drama.mlx.contracts import Chunk as DramaChunk
from voicelab_inference.drama.mlx.runtime import DramaMLX
from voicelab_inference.kitten.torch.contracts import Chunk as KittenChunk
from voicelab_inference.kitten.torch.runtime import Audio, KittenTorch
from voicelab_inference.maya.mlx.contracts import Chunk as MayaChunk
from voicelab_inference.maya.mlx.runtime import MayaMLX
from voicelab_inference.omnivoice.torch.runtime import OmniVoiceTorch
from voicelab_models.audio import ReferenceAudio
from voicelab_models.drama.model import PreviousTake, Reference
from voicelab_models.drama.model import Request as DramaRequest
from voicelab_models.kitten.model import Request as KittenRequest
from voicelab_models.maya.model import Request as MayaRequest
from voicelab_models.omnivoice.model import Request as OmniRequest


class FakeKitten:
    def __init__(self, *, eos: bool = True) -> None:
        self.closed = False
        self.released = 0
        self.eos = eos

    def chunks(self, request: KittenRequest) -> Generator[KittenChunk]:
        try:
            yield KittenChunk(np.zeros(240, dtype=np.float32), 1, self.eos)
            yield KittenChunk(np.zeros(240, dtype=np.float32), 1, True)
        finally:
            self.released += 1

    def close(self) -> None:
        self.closed = True


class FakeOmni:
    def __init__(self) -> None:
        self.closed = False

    def generate(self, request: OmniRequest) -> Audio:
        return np.zeros(240, dtype=np.float32)

    def close(self) -> None:
        self.closed = True


class ExpressiveRuntimeTests(unittest.TestCase):
    def test_request_limits_are_checked_before_loading(self) -> None:
        for request_type in (DramaRequest, MayaRequest, OmniRequest, KittenRequest):
            with self.subTest(request=request_type):
                with self.assertRaises(ValidationError):
                    request_type(text=" \t\n")
                with self.assertRaises(ValidationError):
                    request_type.model_validate({"text": "hello", "unknown": True})
        with self.assertRaises(ValidationError):
            OmniRequest(text="Hello", speed=float("nan"))
        with self.assertRaises(ValidationError):
            KittenRequest(text="Hello", max_tokens=0)

    def test_kitten_natural_completion_and_cancel_cleanup(self) -> None:
        backend = FakeKitten()
        runtime = KittenTorch(loader=lambda device: backend)
        metrics = runtime.synthesize(KittenRequest(text="Hello"), lambda audio: None)
        self.assertEqual(metrics.completion_state, "complete")
        self.assertTrue(metrics.eos)
        self.assertEqual(metrics.frames, 2)
        self.assertEqual(metrics.audio_seconds, 0.02)

        def cancel(audio: Audio) -> None:
            raise InterruptedError("cancelled")

        with self.assertRaises(InterruptedError):
            runtime.synthesize(KittenRequest(text="Hello"), cancel)
        self.assertEqual(backend.released, 2)
        runtime.close()
        self.assertTrue(backend.closed)

    def test_omni_full_output_diffusion_completion(self) -> None:
        backend = FakeOmni()
        runtime = OmniVoiceTorch(loader=lambda device: backend)
        chunks: list[Audio] = []
        metrics = runtime.synthesize(OmniRequest(text="Hello"), chunks.append)
        self.assertEqual(metrics.completion_state, "complete")
        self.assertIsNone(metrics.eos)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(metrics.first_pcm_seconds, metrics.generation_seconds)
        self.assertIsNone(metrics.frames)
        runtime.close()
        self.assertTrue(backend.closed)


class FakeMaya:
    def __init__(self, *, eos: bool, final: bool = True) -> None:
        self.eos = eos
        self.final = final
        self.released = False
        self.closed = False

    def chunks(self, request: MayaRequest) -> Generator[MayaChunk]:
        try:
            yield MayaChunk(np.zeros(1920, np.float32), 1, None, False)
            yield MayaChunk(np.zeros(1920, np.float32), 1, self.eos, self.final)
        finally:
            self.released = True

    def close(self) -> None:
        self.closed = True


class FakeDrama:
    def __init__(self, *, eos: bool) -> None:
        self.eos = eos
        self.released = False
        self.closed = False

    def chunks(self, request: DramaRequest) -> Generator[DramaChunk]:
        try:
            yield DramaChunk(np.zeros(2048, np.float32), 1, None, False)
            # A terminal marker can have no PCM when the last chunk was full.
            yield DramaChunk(np.zeros(0, np.float32), 0, self.eos, True)
        finally:
            self.released = True

    def close(self) -> None:
        self.closed = True


class LocalMLXRuntimeTests(unittest.TestCase):
    def test_maya_complete_and_capped_output(self) -> None:
        for eos in (True, False):
            backend = FakeMaya(eos=eos)
            runtime = MayaMLX(loader=lambda backend=backend: backend)
            chunks: list[Audio] = []
            result = runtime.synthesize(MayaRequest(text="Hello"), chunks.append)
            self.assertEqual(result.eos, eos)
            self.assertEqual(result.frames, 2)
            self.assertEqual(result.audio_seconds, 0.16)
            self.assertEqual(len(chunks), 2)
            self.assertTrue(backend.released)
            runtime.close()
            self.assertTrue(backend.closed)

    def test_maya_requires_a_terminal_marker(self) -> None:
        backend = FakeMaya(eos=True, final=False)
        runtime = MayaMLX(loader=lambda: backend)
        delivered: list[Audio] = []
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            runtime.synthesize(MayaRequest(text="Hello"), delivered.append)
        self.assertEqual(len(delivered), 2)
        self.assertTrue(backend.released)

    def test_maya_cancellation_releases_generator(self) -> None:
        backend = FakeMaya(eos=True)
        runtime = MayaMLX(loader=lambda: backend)

        def cancel(audio: Audio) -> None:
            raise InterruptedError("cancelled")

        with self.assertRaises(InterruptedError):
            runtime.synthesize(MayaRequest(text="Hello"), cancel)
        self.assertTrue(backend.released)
        runtime.close()

    def test_drama_empty_final_marker_and_cap_status(self) -> None:
        for eos in (True, False):
            backend = FakeDrama(eos=eos)
            runtime = DramaMLX(loader=lambda backend=backend: backend)
            chunks: list[Audio] = []
            result = runtime.synthesize(DramaRequest(text="Hello"), chunks.append)
            self.assertEqual(result.eos, eos)
            self.assertEqual(len(chunks), 1)
            self.assertEqual(result.frames, 1)
            self.assertAlmostEqual(result.audio_seconds, 2048 / 44100)
            runtime.close()
            self.assertTrue(backend.closed)

    def test_native_controls_validate_before_model_loading(self) -> None:
        with self.assertRaises(ValidationError):
            MayaRequest.model_validate({"text": "Hello", "voice": "unknown"})
        for text in ("<|im_end|>", "<|speaker:16|> Hello", "<|speaker:01|> Hello"):
            with self.assertRaises(ValidationError):
                DramaRequest(text=text)


def reference_audio() -> ReferenceAudio:
    data = io.BytesIO()
    with wave.open(data, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(bytes(9600))
    return ReferenceAudio(wav_base64=base64.b64encode(data.getvalue()).decode("ascii"))


class ExpressiveConditioningTests(unittest.TestCase):
    def test_reference_transcripts_and_modes_are_validated(self) -> None:
        audio = reference_audio()
        for request in (KittenRequest, OmniRequest):
            with self.assertRaises(ValidationError):
                request(text="hello", ref_audio=audio)
            with self.assertRaises(ValidationError):
                request(text="hello", ref_text="reference")
            self.assertIsNotNone(
                request(text="hello", ref_audio=audio, ref_text="reference").ref_audio
            )
        with self.assertRaises(ValidationError):
            KittenRequest(
                text="hello", ref_audio=audio, ref_text="reference", voice="Bella"
            )
        with self.assertRaises(ValidationError):
            OmniRequest(
                text="hello",
                ref_audio=audio,
                ref_text="reference",
                instruction="Another voice",
            )
        with self.assertRaises(ValidationError):
            MayaRequest(text="hello", ref_audio=audio, conditioning="icl")
        self.assertIsNotNone(
            MayaRequest(text="hello", ref_audio=audio, conditioning="spk").ref_audio
        )
        for mode in ("icl", "icl_spk"):
            self.assertIsNotNone(
                MayaRequest.model_validate(
                    {
                        "text": "hello",
                        "ref_audio": audio,
                        "ref_text": "reference",
                        "conditioning": mode,
                    }
                ).ref_audio
            )
        with self.assertRaises(ValidationError):
            DramaRequest.model_validate(
                {"text": "hello", "references": [{"audio": audio, "text": " "}]}
            )

    def test_a_capped_kitten_sentence_stays_incomplete(self) -> None:
        backend = FakeKitten(eos=False)
        runtime = KittenTorch(loader=lambda device: backend)
        metrics = runtime.synthesize(KittenRequest(text="Hello"), lambda audio: None)
        self.assertEqual(metrics.completion_state, "incomplete")

    def test_missing_normalizer_fails_before_loading(self) -> None:
        loaded = False

        def load(device: str) -> FakeOmni:
            nonlocal loaded
            loaded = True
            return FakeOmni()

        runtime = OmniVoiceTorch(loader=load)
        with (
            patch(
                "voicelab_inference.omnivoice.torch.runtime.find_spec",
                return_value=None,
            ),
            self.assertRaisesRegex(RuntimeError, "normalization"),
        ):
            runtime.synthesize(
                OmniRequest(text="42", normalize=True), lambda audio: None
            )
        self.assertFalse(loaded)


class DramaContinuationTests(unittest.TestCase):
    def test_prior_takes_have_separate_bounded_history(self) -> None:
        audio = reference_audio()
        previous = PreviousTake(audio=audio, text="Earlier words")
        request = DramaRequest(text="Edited words", previous_takes=[previous])
        self.assertEqual(request.previous_takes[0].text, "Earlier words")
        self.assertEqual(request.references, [])
        with self.assertRaises(ValidationError):
            DramaRequest(text="Edited", previous_takes=[previous] * 3)
        with self.assertRaises(ValidationError):
            DramaRequest(
                text="Edited",
                references=[Reference(audio=audio, text="Voice")] * 4,
                previous_takes=[previous],
            )
        with self.assertRaises(ValidationError):
            DramaRequest(
                text="Edited",
                previous_takes=[PreviousTake(audio=audio, text="<|im_end|>")],
            )

    def test_manifest_preserves_order_and_resolves_relative_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prior.wav").write_bytes(reference_audio().wav_bytes())
            manifest = root / "takes.json"
            manifest.write_text(
                json.dumps(
                    [
                        {"audio": "prior.wav", "text": "Wednesday"},
                        {"audio": "prior.wav", "text": "Thursday"},
                    ]
                )
            )
            takes = load_previous_takes(manifest)
            self.assertEqual([take.text for take in takes], ["Wednesday", "Thursday"])
            self.assertEqual(
                takes[0].audio.wav_bytes(), (root / "prior.wav").read_bytes()
            )
            manifest.write_text(
                json.dumps([{"audio": "missing.wav", "text": "Too many"}] * 3)
            )
            with self.assertRaisesRegex(ValueError, "one or two"):
                load_previous_takes(manifest)
