"""Actual MLX state restoration across successful, failed and repeated cleanup."""

import unittest

import mlx.core as mx

from voicelab_inference.nemotron.mlx.sampling import SamplingState, preserved


class SamplingTests(unittest.TestCase):
    def test_sampling_context_restores_after_failure(self) -> None:
        mx.set_default_device(mx.cpu)
        mx.random.seed(37)
        expected = mx.random.uniform(shape=(4,))
        mx.eval(expected)
        mx.random.seed(37)
        with self.assertRaisesRegex(ValueError, "test failure"), preserved():
            mx.random.seed(123)
            mx.eval(mx.random.uniform(shape=(10,)))
            raise ValueError("test failure")
        actual = mx.random.uniform(shape=(4,))
        mx.eval(actual)
        self.assertTrue(bool(mx.array_equal(actual, expected)))

    def test_stream_restore_is_idempotent(self) -> None:
        mx.set_default_device(mx.cpu)
        mx.random.seed(18)
        expected = mx.random.uniform(shape=(4,))
        mx.eval(expected)
        mx.random.seed(18)
        saved = SamplingState()
        mx.random.seed(999)
        saved.restore()
        actual = mx.random.uniform(shape=(4,))
        mx.eval(actual)
        saved.restore()
        second = mx.random.uniform(shape=(4,))
        mx.eval(second)
        self.assertTrue(bool(mx.array_equal(actual, expected)))
        self.assertFalse(bool(mx.array_equal(second, expected)))
