"""Decoded sentence audio and the native token-loop stop evidence."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type Audio = NDArray[np.float32]


@dataclass(frozen=True)
class Chunk:
    audio: Audio
    frames: int
    eos: bool
