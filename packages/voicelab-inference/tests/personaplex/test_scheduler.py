"""Native MLX scheduler checks without model weights or network access."""

import unittest
from dataclasses import replace

import mlx.core as mx
from mlx_audio.sts.models.moshi.lm import Lm

from voicelab_inference.personaplex.mlx.network import PersonaNetwork, configuration
from voicelab_inference.personaplex.mlx.scheduler import Scheduler


class Network:
    def __init__(self) -> None:
        self.forced: list[mx.array] = []
        self.provided: list[mx.array] = []
        self.texts: list[mx.array] = []
        self.inputs: list[mx.array] = []
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def embed(self, tokens: mx.array) -> mx.array:
        self.inputs.append(mx.array(tokens))
        return mx.zeros((1, 1, 1))

    def forward(self, embeddings: mx.array) -> tuple[mx.array, mx.array]:
        return embeddings, mx.zeros((1, 1, 8))

    def text(self, logits: mx.array) -> mx.array:
        return mx.array([[7]], dtype=mx.int32)

    def depth(
        self, hidden: mx.array, text: mx.array, forced: mx.array, provided: mx.array
    ) -> mx.array:
        self.forced.append(mx.array(forced))
        self.provided.append(mx.array(provided))
        self.texts.append(mx.array(text))
        return mx.arange(100, 116, dtype=mx.int32).reshape(1, 16, 1)


class SchedulerTests(unittest.TestCase):
    def test_configuration_preserves_all_depth_conditioning(self) -> None:
        config = configuration()
        self.assertEqual(config.depformer.num_slices, 16)
        self.assertEqual(config.depformer.transformer.context, 16)
        self.assertEqual(config.transformer.dim_feedforward, 16896)
        self.assertEqual(config.depformer.transformer.dim_feedforward, 4224)

    def test_delay_alignment_survives_ring_wrap(self) -> None:
        network = Network()
        scheduler = Scheduler(network)
        for frame in range(12):
            user = mx.full((1, 8, 1), frame, dtype=mx.int32)
            output = scheduler.step(user)
            if frame < 2:
                self.assertIsNone(output)
            else:
                if output is None:
                    self.fail("Missing aligned output")
                self.assertEqual(output[0, :, 0].tolist(), [7, *range(100, 108)])
            if frame >= 2:
                # Input's first codebook is current; delayed input uses prior frame.
                forced = network.forced[-1][0, 8:].tolist()
                self.assertEqual(forced, [frame, *([frame - 1] * 7)])
                self.assertTrue(bool(mx.all(network.provided[-1][0, 8:]).item()))
        self.assertEqual(network.reset_count, 1)

    def test_prompt_forces_text_and_audio_before_depth_sampling(self) -> None:
        network = Network()
        scheduler = Scheduler(network)
        for frame in range(4):
            scheduler.step(
                mx.full((1, 8, 1), 40 + frame, dtype=mx.int32),
                mx.full((1, 8, 1), 60 + frame, dtype=mx.int32),
                20 + frame,
            )
        self.assertEqual(network.texts[-1].item(), 23)
        self.assertEqual(network.forced[-1][0, :8].tolist(), [63, *([62] * 7)])
        self.assertTrue(bool(mx.all(network.provided[-1]).item()))

    def test_native_temporal_and_depth_graph_with_small_weights(self) -> None:
        baseline = configuration()
        small = replace(
            baseline,
            transformer=replace(
                baseline.transformer,
                d_model=16,
                num_heads=2,
                num_layers=1,
                dim_feedforward=64,
            ),
            depformer=replace(
                baseline.depformer,
                transformer=replace(
                    baseline.depformer.transformer,
                    d_model=16,
                    num_heads=2,
                    num_layers=1,
                    dim_feedforward=64,
                ),
            ),
        )
        model = Lm(small)
        network = PersonaNetwork(model)
        scheduler = Scheduler(network)
        output = None
        for _ in range(4):
            output = scheduler.step(mx.zeros((1, 8, 1), dtype=mx.int32))
        if output is None:
            self.fail("Native network produced no aligned frame")
        self.assertEqual(output.shape, (1, 9, 1))
        self.assertTrue(bool(mx.all(output[:, 1:] < 2048).item()))
        restarted = Scheduler(PersonaNetwork(model))
        repeated = None
        for _ in range(4):
            repeated = restarted.step(mx.zeros((1, 8, 1), dtype=mx.int32))
        if repeated is None:
            self.fail("Restarted network produced no frame")
        self.assertTrue(bool(mx.array_equal(output, repeated).item()))

    def test_invalid_prompt_shape_fails_before_network_work(self) -> None:
        network = Network()
        scheduler = Scheduler(network)
        with self.assertRaises(ValueError):
            scheduler.prompt(mx.zeros((1, 4096)), mx.zeros((1, 17, 4)), ())
        self.assertEqual(network.inputs, [])


if __name__ == "__main__":
    unittest.main()
