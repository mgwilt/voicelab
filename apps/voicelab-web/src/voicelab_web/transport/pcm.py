"""Convert validated backend audio to the site's PCM16 wire format."""

import numpy as np
import numpy.typing as npt

type Audio = npt.NDArray[np.float32]


def pcm16(samples: Audio) -> bytes:
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("Expected nonempty finite mono audio")
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
