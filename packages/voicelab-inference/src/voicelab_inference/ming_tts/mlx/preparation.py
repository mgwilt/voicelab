"""Explicit reference-tail conditioning; the original recording stays unchanged."""

import numpy as np
from numpy.typing import NDArray


def trim_reference_tail(
    audio: NDArray[np.float32], sample_rate: int
) -> NDArray[np.float32]:
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Reference must contain finite mono audio")
    threshold = max(1e-4, float(np.max(np.abs(audio))) * 0.01)
    active = np.flatnonzero(np.abs(audio) > threshold)
    if not active.size:
        raise ValueError("Reference contains no usable speech amplitude")
    end = max(int(sample_rate * 0.4), int(active[-1]) + int(sample_rate * 0.05) + 1)
    return audio[: min(audio.size, end)]
