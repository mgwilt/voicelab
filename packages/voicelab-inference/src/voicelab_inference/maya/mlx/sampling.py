"""Preserve the current thread's MLX random key for one inference request."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Protocol, runtime_checkable

import mlx.core as mx


@runtime_checkable
class KeyState(Protocol):
    def __getitem__(self, index: int) -> object: ...


def _key(state: KeyState) -> mx.array:
    key = state[0]
    if not isinstance(key, mx.array) or key.shape != (2,) or key.dtype != mx.uint32:
        raise TypeError("Unsupported MLX random state key")
    return key


@contextmanager
def seeded(seed: int) -> Generator[None]:
    state: object = mx.random.state
    if not isinstance(state, KeyState):
        raise TypeError("Unsupported MLX random state container")
    saved = mx.array(_key(state))
    mx.eval(saved)
    mx.random.seed(seed)
    try:
        yield
    finally:
        try:
            mx.synchronize()
        finally:
            # MLX 0.32 exposes a thread-local container with mutable array keys.
            _key(state)[:] = saved
