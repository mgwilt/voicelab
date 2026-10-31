"""Official posterior and acoustic-reference boundary regressions."""

import unittest

import mlx.core as mx
import numpy as np

from voicelab_inference.ming_audio.mlx.reference import (
    posterior_sample,
    trim_reference_tail,
)


class ReferenceTests(unittest.TestCase):
    def test_oobleck_uses_softplus_scale_not_log_variance(self) -> None:
        parameters = mx.array([[[1.0, -2.0, 0.0, 2.0]]])
        noise = mx.array([[[1.0, -1.0]]])
        actual = posterior_sample(parameters, noise)
        expected = np.array([1.0, -2.0]) + (np.logaddexp([0.0, 2.0], 0) + 1e-4) * [
            1,
            -1,
        ]
        np.testing.assert_allclose(np.asarray(actual).reshape(-1), expected, rtol=1e-6)
        with self.assertRaises(ValueError):
            posterior_sample(parameters, mx.zeros((1, 1, 3)))

    def test_reference_tail_keeps_margin_and_rejects_silence(self) -> None:
        samples = np.zeros(16000, dtype=np.float32)
        samples[:8000] = 0.1
        result = trim_reference_tail(samples, 16000)
        self.assertEqual(result.size, 8800)
        np.testing.assert_array_equal(result, samples[:8800])
        with self.assertRaises(ValueError):
            trim_reference_tail(np.zeros(16000, dtype=np.float32), 16000)


if __name__ == "__main__":
    unittest.main()
