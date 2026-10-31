"""Request-local seeded sampling adapted from MLX-Audio; see NOTICE/LICENSE."""

from collections.abc import Generator

import mlx.core as mx
from mlx_audio.lm.models.cache import KVCache
from mlx_audio.tts.models.bailingmm.bailingmm import (
    MING3_MAX_CACHE_LEN,
    MING3_TARGET_LANGUAGES,
    ming3_instruction,
)
from mlx_audio.tts.models.dense.dense import Model

from voicelab_models.ming_tts.model import Request


def sample(
    model: Model,
    request: Request,
    *,
    text: str,
    prompt_latent: mx.array | None = None,
    speaker: mx.array | None = None,
    segment: int = 0,
) -> Generator[tuple[mx.array, bool]]:
    prompt = "Please generate speech based on the following description.\n"
    language = MING3_TARGET_LANGUAGES.get(request.language)
    if language is not None:
        prompt += f"Target language: {language}.\n"
    instruction = request.instruction
    if any(
        value is not None
        for value in (
            request.voice,
            request.style,
            request.emotion,
            request.dialect,
            request.speed,
            request.pitch,
            request.volume,
        )
    ):
        instruction = ming3_instruction(
            ip=request.voice,
            style=request.style,
            emotion=request.emotion,
            dialect=request.dialect,
            speed=request.speed,
            pitch=request.pitch,
            volume=request.volume,
        )
    speakers = model._speaker_embeddings(
        speaker, use_zero_spk_emb=speaker is None and instruction is not None
    )
    _, embeddings = model._prepare_input_embed(
        prompt=prompt,
        text=text,
        instruction=instruction,
        prompt_latent=prompt_latent,
        prompt_text=request.ref_text,
        spk_emb=speakers,
    )
    if not model.ming3 or not model.flowloss.cfm.ming3_sampling:
        raise ValueError("This sampler requires the Ming 3 dense architecture")
    budget = (MING3_MAX_CACHE_LEN - embeddings.shape[1]) // model.patch_size
    if budget < 1:
        raise ValueError("Ming text and instruction exceed the context window")
    cache = [KVCache() for _ in range(model.llm_args.num_hidden_layers)]
    key = mx.random.key((request.seed + segment) % 2**32)
    history: mx.array | None = None
    for step in range(min(request.max_frames, budget)):
        hidden = model._forward_llm_hidden(inputs_embeds=embeddings, cache=cache)
        condition = hidden[:, -1:, :]
        if history is None:
            history = mx.zeros(
                (1, model.history_patch_size, model.latent_dim), dtype=condition.dtype
            )
            if prompt_latent is not None:
                reference = prompt_latent.astype(condition.dtype)
                if reference.shape[1] >= model.history_patch_size:
                    history = reference[:, -model.history_patch_size :, :]
                else:
                    history = mx.concatenate(
                        [
                            history[
                                :, : model.history_patch_size - reference.shape[1], :
                            ],
                            reference,
                        ],
                        axis=1,
                    )
        key, initial_key, sde_key = mx.random.split(key, 3)
        noise = mx.random.normal(
            (1, model.patch_size, model.flowloss.z_channels),
            dtype=condition.dtype,
            key=initial_key,
        )
        sde_noise = mx.random.normal(
            (request.flow_steps, *noise.shape), dtype=condition.dtype, key=sde_key
        )
        latent, _ = model.flowloss.cfm.sample(
            noise=noise,
            c=condition,
            latent_history=history,
            cfg_scale=request.cfg_scale,
            patch_size=model.patch_size,
            sigma=request.sigma,
            temperature=request.temperature,
            steps=request.flow_steps,
            sde_noise=sde_noise,
        )
        probability = mx.softmax(model.stop_head(condition), axis=-1)[0, 0, 1]
        last = bool(probability > 0.5 and step > 3)
        yield latent, last
        if last:
            return
        embeddings = model.linear_proj_audio(latent)
        history = mx.concatenate([history[:, model.patch_size :, :], latent], axis=1)
