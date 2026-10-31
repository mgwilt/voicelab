"""Use the official Oobleck posterior for Ming acoustic continuation prompts."""

import mlx.core as mx
import numpy as np
from mlx_audio.sts.models.ming_v3_audio.talker import Talker
from numpy.typing import NDArray


def posterior_sample(parameters: mx.array, noise: mx.array) -> mx.array:
    """Ming stores a raw softplus scale, not a log variance."""
    mean, scale = mx.split(parameters, 2, axis=-1)
    if noise.shape != mean.shape:
        raise ValueError("Reference posterior noise must match the latent shape.")
    deviation = mx.logaddexp(scale, mx.zeros_like(scale)) + 1e-4
    return mean + deviation * noise


def encode_reference(talker: Talker, waveform: mx.array) -> mx.array:
    padded = talker._pad_prompt_waveform(waveform)
    parameters = talker.audio.encoder(padded)
    if not isinstance(parameters, mx.array) or parameters.ndim != 3:
        raise TypeError("Reference encoder must return [batch, frames, parameters].")
    if parameters.shape[-1] != 2 * talker.latent_dim:
        raise ValueError("Reference posterior parameters have the wrong width.")
    noise = mx.random.normal(
        (*parameters.shape[:-1], talker.latent_dim), dtype=parameters.dtype
    )
    latent = posterior_sample(parameters, noise)
    mx.eval(latent)
    if not bool(mx.all(mx.isfinite(latent)).item()):
        raise ValueError("Reference posterior produced non-finite latents.")
    return latent


def trim_reference_tail(
    waveform: NDArray[np.float32], sample_rate: int
) -> NDArray[np.float32]:
    """Retain a 50 ms tail after samples above -40 dB of the reference peak."""
    peak = float(np.max(np.abs(waveform)))
    active = np.flatnonzero(np.abs(waveform) > max(0.0001, peak * 0.01))
    if not active.size:
        raise ValueError("A voice reference must contain audible speech.")
    end = max(int(sample_rate * 0.4), int(active[-1]) + 1 + int(sample_rate * 0.05))
    return waveform[:end]
