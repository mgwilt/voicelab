"""Numerical and lifecycle regressions adapted from the source project."""

import unittest

import numpy as np
import torch
from torch import Tensor
from transformers import CsmDepthDecoderConfig, MimiConfig
from transformers.models.csm.modeling_csm import CsmDepthDecoderForCausalLM
from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiConvTranspose1d

from voicelab_inference.sesame.torch.audio import Audio
from voicelab_inference.sesame.torch.decoding.depth import direct_depth_generate
from voicelab_inference.sesame.torch.decoding.mimi import ConvolutionHistory
from voicelab_inference.sesame.torch.decoding.stream import AudioStreamer
from voicelab_models.sesame.model import SynthesisRequest


class FakeCodec:
    def decode(self, codes: Tensor) -> Audio:
        return np.repeat(np.asarray(codes[0, :, 0], dtype=np.float32), 1920)

    def close(self) -> None:
        pass


class DecodingTests(unittest.TestCase):
    def test_direct_matches_reference(self) -> None:
        for sample in (False, True):
            with self.subTest(sample=sample):
                torch.manual_seed(9)
                config = CsmDepthDecoderConfig(
                    hidden_size=32,
                    intermediate_size=64,
                    backbone_hidden_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=8,
                    vocab_size=64,
                    num_codebooks=4,
                    pad_token_id=63,
                    bos_token_id=0,
                    eos_token_id=None,
                )
                model = CsmDepthDecoderForCausalLM(config).eval()
                settings = model.generation_config
                assert settings is not None
                settings.do_sample = sample
                settings.temperature = 0.9 if sample else None
                settings.top_k = 5 if sample else None
                settings.min_new_tokens = 3
                settings.max_new_tokens = 3
                ids = torch.tensor([[0, 12]])
                hidden = torch.randn(1, 64)
                with torch.inference_mode():
                    torch.manual_seed(42)
                    expected = model.generate(
                        input_ids=ids, backbone_last_hidden_state=hidden
                    )
                    torch.manual_seed(42)
                    actual = direct_depth_generate(model, ids, hidden)
                assert isinstance(expected, Tensor)
                self.assertTrue(torch.equal(expected, actual))

    def test_causal_history_and_restoration(self) -> None:
        torch.manual_seed(3)
        config = MimiConfig(use_causal_conv=True, trim_right_ratio=1.0)
        network = torch.nn.Sequential(
            MimiConvTranspose1d(config, 2, 3, kernel_size=4, stride=2, bias=True),
            MimiConv1d(config, 3, 2, kernel_size=7, bias=True),
            MimiConvTranspose1d(config, 2, 1, kernel_size=8, stride=4, bias=True),
            MimiConv1d(config, 1, 1, kernel_size=3, bias=True),
        ).eval()
        values = torch.randn(1, 2, 19)
        with torch.inference_mode():
            expected = network(values)
            state = ConvolutionHistory(network)
            try:
                streamed = torch.cat(
                    [network(values[..., i : i + 4]) for i in range(0, 19, 4)], dim=-1
                )
            finally:
                state.close()
            self.assertTrue(torch.allclose(expected, streamed, atol=1e-6))
            self.assertTrue(torch.equal(network(values), expected))

    def test_flush_eos_metrics_and_consumer_failure(self) -> None:
        chunks: list[Audio] = []
        stream = AudioStreamer(
            FakeCodec(),
            chunks.append,
            device="cpu",
            codebooks=2,
            eos_token=0,
            max_frames=12,
        )
        stream.put(torch.tensor([[123, 456, 789]]))
        for value in range(1, 10):
            stream.put(torch.tensor([[value, value]]))
        stream.put(torch.zeros(1, 2, dtype=torch.long))
        stream.end()
        stream.end()
        self.assertTrue(stream.saw_eos)
        self.assertTrue(
            np.array_equal(np.concatenate(chunks), np.repeat(np.arange(1, 10), 1920))
        )
        self.assertEqual(len(stream.metrics().arrivals), 3)
        self.assertEqual(stream.metrics(), stream.metrics())

        def cancelled(audio: Audio) -> None:
            raise BrokenPipeError("Client left")

        stream = AudioStreamer(
            FakeCodec(), cancelled, device="cpu", codebooks=2, eos_token=0, max_frames=8
        )
        stream.put(torch.tensor([[123]]))
        for _ in range(3):
            stream.put(torch.tensor([[1, 1]]))
        with self.assertRaises(BrokenPipeError):
            stream.put(torch.tensor([[1, 1]]))

    def test_validation_and_partial_eos(self) -> None:
        for payload in (
            {"text": " "},
            {"text": "hi", "max_frames": 1501},
            {"text": "hi", "seed": "42"},
        ):
            with self.assertRaises(ValueError):
                SynthesisRequest.model_validate(payload)
        stream = AudioStreamer(
            FakeCodec(),
            lambda _: None,
            device="cpu",
            codebooks=2,
            eos_token=0,
            max_frames=4,
        )
        stream.put(torch.tensor([[123]]))
        for _ in range(4):
            stream.put(torch.tensor([[0, 1]]))
        stream.end()
        self.assertFalse(stream.metrics().eos)
        with self.assertRaises(RuntimeError):
            stream.put(torch.tensor([[1, 1]]))

    def test_empty_generation_is_not_success(self) -> None:
        stream = AudioStreamer(
            FakeCodec(),
            lambda _: None,
            device="cpu",
            codebooks=2,
            eos_token=0,
            max_frames=4,
        )
        stream.put(torch.tensor([[123]]))
        stream.put(torch.zeros(1, 2, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "no audio"):
            stream.end()
