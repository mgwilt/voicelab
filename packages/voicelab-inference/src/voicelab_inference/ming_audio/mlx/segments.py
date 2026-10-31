"""Aggregate bounded speech fragments without losing partial-output evidence."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Segment:
    audio: NDArray[np.float32]
    steps: int
    completed: bool


def combine(
    fragments: Sequence[str],
    step_budget: int,
    generate: Callable[[str, int], Segment],
    should_cancel: Callable[[], bool],
) -> tuple[NDArray[np.float32], bool]:
    if not fragments or step_budget < 1:
        raise ValueError("Speech requires text fragments and a positive step budget.")
    remaining = step_budget
    parts: list[NDArray[np.float32]] = []
    for fragment in fragments:
        if should_cancel():
            raise InterruptedError("Audio conversation cancelled")
        if remaining == 0:
            return np.concatenate(parts), False
        segment = generate(fragment, remaining)
        if not 0 < segment.steps <= remaining:
            raise ValueError("Talker exceeded its acoustic step budget.")
        remaining -= segment.steps
        parts.append(segment.audio)
        if not segment.completed:
            return np.concatenate(parts), False
    return np.concatenate(parts), True
