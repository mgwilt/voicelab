"""Request seeding must not affect other experiments or unused devices."""

import unittest
from unittest.mock import patch

import torch

from voicelab_inference.sesame.torch.random import random_state


class RandomStateTests(unittest.TestCase):
    def test_cpu_restores_rng_after_failure_without_seeding_other_devices(self) -> None:
        before = torch.get_rng_state()
        with (
            patch.object(torch.mps, "manual_seed") as mps_seed,
            patch.object(torch.cuda, "manual_seed_all") as cuda_seed,
        ):
            with (
                self.assertRaisesRegex(RuntimeError, "cancel"),
                random_state("cpu", 19),
            ):
                torch.rand(3)
                raise RuntimeError("cancel")
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            mps_seed.assert_not_called()
            cuda_seed.assert_not_called()

    def test_repeated_requests_use_same_random_sequence(self) -> None:
        with random_state("cpu", 19):
            first = torch.rand(8)
        with random_state("cpu", 19):
            second = torch.rand(8)
        self.assertTrue(torch.equal(first, second))
