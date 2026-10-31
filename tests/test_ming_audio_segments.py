"""Long-form speech preserves ordered PCM and aggregate completion limits."""

import unittest

import numpy as np

from voicelab_inference.ming_audio.mlx.segments import Segment, combine


def never_cancel() -> bool:
    return False


class SegmentTests(unittest.TestCase):
    def test_multiple_completed_fragments_preserve_order_and_share_budget(self) -> None:
        calls: list[tuple[str, int]] = []

        def generate(text: str, budget: int) -> Segment:
            calls.append((text, budget))
            value = 0.1 if text == "first" else 0.2
            return Segment(np.full(3, value, dtype=np.float32), 2, True)

        audio, completed = combine(("first", "second"), 5, generate, never_cancel)
        self.assertTrue(completed)
        self.assertEqual(calls, [("first", 5), ("second", 3)])
        np.testing.assert_array_equal(
            audio, np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2], dtype=np.float32)
        )

    def test_boundary_cap_does_not_claim_unspoken_fragment_completed(self) -> None:
        calls: list[str] = []

        def generate(text: str, budget: int) -> Segment:
            calls.append(text)
            return Segment(np.ones(3, dtype=np.float32), budget, True)

        audio, completed = combine(("first", "unspoken"), 2, generate, never_cancel)
        self.assertFalse(completed)
        self.assertEqual(calls, ["first"])
        self.assertEqual(audio.size, 3)

    def test_incomplete_fragment_retains_partial_pcm_and_stops(self) -> None:
        calls: list[str] = []

        def generate(text: str, budget: int) -> Segment:
            del budget
            calls.append(text)
            return Segment(np.ones(3, dtype=np.float32), 1, text == "first")

        audio, completed = combine(
            ("first", "partial", "unspoken"), 20, generate, never_cancel
        )
        self.assertFalse(completed)
        self.assertEqual(calls, ["first", "partial"])
        self.assertEqual(audio.size, 6)

    def test_cancellation_between_fragments_does_not_start_another(self) -> None:
        calls: list[str] = []

        def generate(text: str, budget: int) -> Segment:
            del budget
            calls.append(text)
            return Segment(np.ones(3, dtype=np.float32), 1, True)

        with self.assertRaises(InterruptedError):
            combine(("first", "unspoken"), 20, generate, lambda: bool(calls))
        self.assertEqual(calls, ["first"])


if __name__ == "__main__":
    unittest.main()
