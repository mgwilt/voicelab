"""Request-local MLX randomness survives normal, failed, and cancelled turns."""

import unittest

import mlx.core as mx
import numpy as np

from voicelab_inference.ming_audio.mlx.sampling import seeded


class SamplingTests(unittest.TestCase):
    def test_repeated_seed_ignores_intervening_random_calls(self) -> None:
        with seeded(17):
            first = np.asarray(mx.random.normal((16,)))
        mx.eval(mx.random.normal((27,)))
        with seeded(17):
            repeated = np.asarray(mx.random.normal((16,)))
        np.testing.assert_array_equal(first, repeated)

    def test_caller_sequence_survives_all_exit_paths(self) -> None:
        for failure in (None, ValueError, InterruptedError):
            with self.subTest(failure=failure):
                mx.random.seed(73)
                expected = np.asarray(mx.random.normal((8,)))
                mx.random.seed(73)
                try:
                    with seeded(11):
                        mx.eval(mx.random.normal((16,)))
                        if failure is not None:
                            raise failure("simulated request exit")
                except ValueError, InterruptedError:
                    pass
                actual = np.asarray(mx.random.normal((8,)))
                np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
