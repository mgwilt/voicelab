"""Keep vendor sampling state local to the owning request or stream lifetime."""

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


class SamplingState:
    def __init__(self) -> None:
        state: object = mx.random.state
        if not isinstance(state, KeyState):
            raise TypeError("Unsupported MLX random state container")
        self._state = state
        self._saved = mx.array(_key(state))
        mx.eval(self._saved)
        self._restored = False

    def restore(self) -> None:
        if not self._restored:
            try:
                mx.synchronize()
            finally:
                _key(self._state)[:] = self._saved
                self._restored = True


@contextmanager
def preserved() -> Generator[None]:
    state = SamplingState()
    try:
        yield
    finally:
        state.restore()
