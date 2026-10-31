"""Sample the released Oobleck posterior with request-local randomness."""

import mlx.core as mx
from mlx_audio.tts.models.dense.dense import Model


def posterior_sample(parameters: mx.array, noise: mx.array) -> mx.array:
    mean, scale = mx.split(parameters, 2, axis=-1)
    if noise.shape != mean.shape:
        raise ValueError("Reference posterior noise must match the latent shape")
    deviation = mx.logaddexp(scale, mx.zeros_like(scale)) + 1e-4
    return mean + deviation * noise


def encode_reference(model: Model, waveform: mx.array, seed: int) -> mx.array:
    padded = model._pad_prompt_waveform(waveform)
    parameters = model.audio.encoder(padded)
    if not isinstance(parameters, mx.array) or parameters.ndim != 3:
        raise ValueError("Reference encoder must return [batch, frames, parameters]")
    if parameters.shape[-1] != 2 * model.latent_dim:
        raise ValueError("Reference posterior has an incompatible latent dimension")
    noise = mx.random.normal(
        (*parameters.shape[:-1], model.latent_dim),
        dtype=parameters.dtype,
        key=mx.random.key(seed),
    )
    latents = posterior_sample(parameters, noise)
    mx.eval(latents)
    if not bool(mx.all(mx.isfinite(latents)).item()):
        raise ValueError("Reference posterior produced non-finite latents")
    return latents
