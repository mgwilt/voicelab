"""Reference audio decoding and scoped files shared across runtime boundaries."""

import io
import math
import wave
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly

from voicelab_models.audio import ReferenceAudio


def reference_pcm(
    reference: ReferenceAudio, target_sample_rate: int
) -> NDArray[np.float32]:
    if not 8000 <= target_sample_rate <= 48000:
        raise ValueError("Unsupported reference target sample rate")
    with wave.open(io.BytesIO(reference.wav_bytes()), "rb") as audio:
        rate = audio.getframerate()
        pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    samples: NDArray[np.float32] = pcm.astype(np.float32) / np.float32(32768)
    if rate != target_sample_rate:
        divisor = math.gcd(rate, target_sample_rate)
        samples = np.asarray(
            resample_poly(samples, target_sample_rate // divisor, rate // divisor),
            dtype=np.float32,
        )
    if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise ValueError("Reference decoding produced invalid audio")
    return samples


@contextmanager
def reference_file(reference: ReferenceAudio | None) -> Generator[str | None]:
    if reference is None:
        yield None
        return
    with TemporaryDirectory(prefix="voicelab-reference-") as directory:
        path = Path(directory) / "reference.wav"
        path.write_bytes(reference.wav_bytes())
        yield str(path)
