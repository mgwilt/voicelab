"""Request-local MLX random state and cancellation."""

from collections.abc import Callable, Generator
from contextlib import contextmanager

import mlx.core as mx


@contextmanager
def seeded(seed: int) -> Generator[None]:
    saved = mx.random.state
    mx.eval(saved)
    mx.random.seed(seed)
    try:
        yield
    finally:
        mx.synchronize()
        mx.random.state = saved


def check_cancel(should_cancel: Callable[[], bool]) -> None:
    if should_cancel():
        raise InterruptedError("Qwen Omni experiment cancelled")


def token_id(value: mx.array) -> int:
    scalar = value.item()
    if not isinstance(scalar, int):
        raise TypeError("Expected an integer token")
    return scalar
