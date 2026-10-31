"""Restore all device generators that the vendor's manual_seed can change."""

from collections.abc import Generator
from contextlib import contextmanager

import torch


@contextmanager
def preserve_random_state() -> Generator[None]:
    cuda_devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    with torch.random.fork_rng(devices=cuda_devices, device_type="cuda"):
        if torch.backends.mps.is_available():
            with torch.random.fork_rng(devices=[0], device_type="mps"):
                yield
        else:
            yield
