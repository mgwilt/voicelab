"""Sampling semantics, stream lifecycle and bounded generation behavior."""

import unittest
from collections.abc import Iterator

import numpy as np
from pydantic import ValidationError

try:
    import mlx.core as mx
except ImportError as exc:
    raise unittest.SkipTest(f"MLX backend tests require a Metal device: {exc}") from exc

from voicelab_inference.breeze.mlx.audio import Audio
from voicelab_inference.breeze.mlx.sampling import (
    Sampler,
    backbone_scores,
    depth_scores,
    guidance,
)
from voicelab_inference.breeze.mlx.stream import synthesize
from voicelab_models.breeze.synthesis import SynthesisRequest


class FakeModel:
    def __init__(self, *, eos: bool = True, empty: bool = False) -> None:
        self.eos = eos
        self.empty = empty
        self.resets = 0

    def reset_codec(self) -> None:
        self.resets += 1

    def decode(self, frames: list[mx.array]) -> mx.array:
        return mx.ones((len(frames) * 1920,), dtype=mx.float32) * 0.25

    def frames(self, request: SynthesisRequest) -> Iterator[mx.array | None]:
        if not self.empty:
            for _ in range(min(4, request.max_frames)):
                yield mx.zeros((16,), dtype=mx.int32)
        if self.eos:
            yield None


class StreamTests(unittest.TestCase):
    def test_complete_audio_and_output_timing(self) -> None:
        chunks: list[Audio] = []
        model = FakeModel()
        result = synthesize(model, SynthesisRequest(text="Hello"), chunks.append)
        self.assertTrue(result.eos)
        self.assertEqual(result.frames, 4)
        self.assertEqual(result.audio_seconds, 0.32)
        self.assertEqual([chunk.size for chunk in chunks], [5760, 1920])
        self.assertEqual(model.resets, 2)
        self.assertEqual(result.output_rtf, result.generation_seconds / 0.32)

    def test_budget_retains_audio_but_is_incomplete(self) -> None:
        model = FakeModel(eos=False)
        chunks: list[Audio] = []
        result = synthesize(
            model, SynthesisRequest(text="Hello", max_frames=1), chunks.append
        )
        self.assertFalse(result.eos)
        self.assertEqual(result.frames, 1)
        self.assertEqual(model.resets, 2)

    def test_cancel_callback_resets_codec(self) -> None:
        model = FakeModel()

        def cancel(_: Audio) -> None:
            raise InterruptedError("Cancelled")

        with self.assertRaises(InterruptedError):
            synthesize(model, SynthesisRequest(text="Hello"), cancel)
        self.assertEqual(model.resets, 2)

    def test_empty_eos_is_failure(self) -> None:
        model = FakeModel(empty=True)
        with self.assertRaisesRegex(RuntimeError, "no audio"):
            synthesize(model, SynthesisRequest(text="Hello"), lambda _: None)
        self.assertEqual(model.resets, 2)


class SamplingTests(unittest.TestCase):
    def test_depth_crossing_token_is_retained(self) -> None:
        logits = mx.log(mx.array([[0.5, 0.3, 0.2]]))
        result = np.array(
            mx.exp(depth_scores(logits, temperature=1, top_k=0, top_p=0.6))
        )
        np.testing.assert_allclose(result, [[0.625, 0.375, 0]], atol=1e-6)

    def test_backbone_warpers_run_before_reserved_mask(self) -> None:
        logits = mx.zeros((1, 2052)).at[0, 2048].add(100)
        scores = backbone_scores(logits, top_k=1)
        with self.assertRaisesRegex(ValueError, "fully filtered"):
            Sampler(0).sample(scores)

    def test_eos_is_only_allowed_in_backbone(self) -> None:
        logits = mx.zeros((1, 2052)).at[0, 2051].add(100)
        self.assertEqual(Sampler(0).sample(backbone_scores(logits)).item(), 2051)
        self.assertTrue(bool(mx.isneginf(depth_scores(logits)[0, 2051]).item()))

    def test_cfg_calculated_in_float32(self) -> None:
        conditional = mx.array([[1.5, 2]], dtype=mx.bfloat16)
        unconditional = mx.array([[0.5, 1]], dtype=mx.bfloat16)
        result = guidance(conditional, unconditional, 4)
        self.assertEqual(result.dtype, mx.float32)
        np.testing.assert_array_equal(np.array(result), [[4.5, 5]])

    def test_request_rejects_blank_and_nonfinite(self) -> None:
        for text, scale in [
            (" ", 4.0),
            ("Hello", float("nan")),
            ("Hello", float("inf")),
        ]:
            with self.assertRaises(ValidationError):
                SynthesisRequest(text=text, cfg_scale=scale)
