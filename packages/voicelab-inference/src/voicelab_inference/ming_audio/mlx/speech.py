"""Native text segmentation with a bounded aggregate Talker step budget."""

from collections.abc import Callable

import mlx.core as mx
import numpy as np
from mlx_audio.sts.models.ming_v3_audio.talker import Talker
from mlx_audio.tts.models.bailingmm.bailingmm import (
    MING3_MAX_FRAGMENT_LENGTH,
    _add_period,
    _split_fragments,
    _trim_trailing_silence,
)
from numpy.typing import NDArray

from voicelab_inference.ming_audio.mlx.conditioning import Conditioning, prepare
from voicelab_inference.ming_audio.mlx.segments import Segment, combine
from voicelab_models.ming_audio.model import OUTPUT_SAMPLE_RATE, Request


def synthesize(
    talker: Talker, text: str, request: Request, should_cancel: Callable[[], bool]
) -> tuple[NDArray[np.float32], bool]:
    conditioning = prepare(talker, request)
    fragments = _split_fragments(_add_period(text), MING3_MAX_FRAGMENT_LENGTH)

    def generate(fragment: str, remaining: int) -> Segment:
        return decode_segment(
            talker, fragment, request, conditioning, remaining, should_cancel
        )

    return combine(fragments, request.max_decode_steps, generate, should_cancel)


def decode_segment(
    talker: Talker,
    text: str,
    request: Request,
    conditioning: Conditioning,
    step_budget: int,
    should_cancel: Callable[[], bool],
) -> Segment:
    sampler = talker.sample(
        prompt="Please generate speech based on the following description.\n",
        text=text,
        instruction=conditioning.instruction,
        prompt_latent=conditioning.latent,
        prompt_text=request.reference_text,
        spk_emb=conditioning.speaker,
        max_decode_steps=step_budget,
        cfg=request.cfg_scale,
        sigma=request.sigma,
        temperature=request.temperature,
        flow_steps=request.flow_steps,
    )
    latents: list[mx.array] = []
    completed = False
    try:
        for latent, last in sampler:
            if should_cancel():
                raise InterruptedError("Audio conversation cancelled")
            if not isinstance(latent, mx.array) or not isinstance(last, bool):
                raise ValueError("Ming Talker returned invalid latent data")
            mx.eval(latent)
            latents.append(latent)
            completed = last
    finally:
        sampler.close()
    if not latents:
        raise RuntimeError("Ming Talker generated no speech")
    if should_cancel():
        raise InterruptedError("Audio conversation cancelled")
    decoded, _, _ = talker.audio.decode(
        mx.concatenate(latents, axis=1), use_cache=False, last_chunk=True
    )
    if not isinstance(decoded, mx.array) or decoded.ndim != 2 or decoded.shape[0] != 1:
        raise ValueError("Ming codec must return mono [1, samples] audio")
    audio = _trim_trailing_silence(decoded[0], OUTPUT_SAMPLE_RATE)
    mx.eval(audio)
    return Segment(np.asarray(audio, dtype=np.float32), len(latents), completed)
