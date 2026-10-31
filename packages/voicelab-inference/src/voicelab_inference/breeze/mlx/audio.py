"""Validated public audio boundary for MLX synthesis."""

import numpy as np
import numpy.typing as npt

type Audio = npt.NDArray[np.float32]


def validate(samples: Audio) -> None:
    if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise ValueError("Expected nonempty finite mono audio")


def pcm16(samples: Audio) -> bytes:
    validate(samples)
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
