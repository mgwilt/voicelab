"""Verify stop evidence at the native token boundary without model weights."""

import unittest

import torch
from kittenml.kittentts2.tokens import TokenMap

from voicelab_inference.kitten.torch.completion import GenerationObserver


class FakeGenerate:
    def __init__(self, output: torch.Tensor) -> None:
        self.output = output

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.output


class KittenCompletionTests(unittest.TestCase):
    def test_raw_eos_and_token_cap_are_distinct(self) -> None:
        tokens = TokenMap(10, 10, 90, 99, 91, 92, 93, 98)
        for suffix, eos in (
            ([10, 11, 99], True),
            ([10, 11, 98], True),
            ([10, 11], False),
        ):
            output = torch.tensor([[1, 2, *suffix]])

            observer = GenerationObserver(FakeGenerate(output), tokens)
            result = observer(torch.tensor([[1, 2]]))
            self.assertIs(result, output)
            stop = observer.take()
            self.assertIsNotNone(stop)
            if stop is None:
                self.fail("Missing stop evidence")
            self.assertEqual(stop.frames, 2)
            self.assertEqual(stop.eos, eos)
            self.assertIsNone(observer.take())

    def test_invalid_output_cannot_be_reported_complete(self) -> None:
        def generate(*args: object, **kwargs: object) -> object:
            return torch.tensor([[1, 2]])

        observer = GenerationObserver(
            generate, TokenMap(10, 10, 90, 99, 91, 92, 93, 98)
        )
        with self.assertRaises(ValueError):
            observer(torch.tensor([[1, 2]]))
        self.assertIsNone(observer.take())
