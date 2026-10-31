"""Seed only the generators used by this request and restore them on every exit."""

from collections.abc import Generator
from contextlib import contextmanager

import torch

from voicelab_models.sesame.model import Device


@contextmanager
def random_state(device: Device, seed: int) -> Generator[None]:
    cpu = torch.get_rng_state()
    selected = (
        torch.mps.get_rng_state()
        if device == "mps"
        else torch.cuda.get_rng_state()
        if device == "cuda"
        else None
    )
    try:
        torch.random.default_generator.manual_seed(seed)
        if device == "mps":
            torch.mps.manual_seed(seed)
        elif device == "cuda":
            torch.cuda.manual_seed(seed)
        yield
    finally:
        try:
            torch.set_rng_state(cpu)
        finally:
            if selected is not None:
                if device == "mps":
                    torch.mps.set_rng_state(selected)
                else:
                    torch.cuda.set_rng_state(selected)
