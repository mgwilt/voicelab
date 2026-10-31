"""Validated mono PCM boundaries for reference and generated speech."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf
from torch import Tensor

from voicelab_models.sesame.model import SAMPLE_RATE

type Audio = npt.NDArray[np.float32]


def validate_audio(samples: Audio) -> Audio:
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("Expected nonempty finite mono audio")
    return samples


def tensor_audio(value: Tensor) -> Audio:
    return validate_audio(
        np.asarray(value.detach().float().cpu().numpy(), dtype=np.float32)
    )


def read_reference(path: Path) -> Audio:
    values, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError("Reference audio must be 24 kHz")
    samples = validate_audio(np.asarray(values.mean(axis=1), dtype=np.float32))
    if samples.size > SAMPLE_RATE * 30:
        raise ValueError("Reference audio must be at most 30 seconds")
    return samples


def pcm16(samples: Audio) -> bytes:
    return (np.clip(validate_audio(samples), -1, 1) * 32767).astype("<i2").tobytes()
