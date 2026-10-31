"""Bounded codec-token generation and causal waveform delivery."""

from collections.abc import Callable

import mlx.core as mx
import numpy as np
from mlx_vlm.models.cache import KVCache
from mlx_vlm.models.qwen3_omni_moe.code2wav import Code2WavModel
from mlx_vlm.models.qwen3_omni_moe.qwen3_omni_moe import Model
from mlx_vlm.sample_utils import make_sampler
from numpy.typing import NDArray

from voicelab_inference.qwen_omni.mlx.conditioning import Conditioning
from voicelab_inference.qwen_omni.mlx.sampling import check_cancel, token_id
from voicelab_models.qwen_omni.model import Request


def waveform(
    codec: Code2WavModel, codes: list[mx.array], start: int
) -> NDArray[np.float32]:
    context = max(0, start - 25)
    packed = mx.stack(codes[context:], axis=1).transpose(0, 2, 1)
    decoded = codec(codes=packed)
    if (
        not isinstance(decoded, mx.array)
        or decoded.ndim != 3
        or decoded.shape[:2] != (1, 1)
    ):
        raise ValueError("Qwen codec must return [1,1,samples]")
    stride = 1
    for ratio in (*codec.config.upsampling_ratios, *codec.config.upsample_rates):
        stride *= ratio
    audio = decoded[0, 0, (start - context) * stride :].astype(mx.float32)
    mx.eval(audio)
    if audio.size != (len(codes) - start) * stride:
        raise ValueError("Qwen codec sample count does not match generated frames")
    return np.asarray(audio, dtype=np.float32)


def generate(
    model: Model,
    conditioning: Conditioning,
    request: Request,
    emit: Callable[[NDArray[np.float32]], None],
    should_cancel: Callable[[], bool],
) -> tuple[bool, int]:
    talker, codec = model.talker, model.code2wav
    if talker is None or codec is None:
        raise ValueError("Checkpoint requires Talker and Code2Wav")
    check_cancel(should_cancel)
    cache = [KVCache() for _ in range(talker.config.text_config.num_hidden_layers)]
    logits, hidden = talker(
        inputs_embeds=conditioning.embeddings, past_key_values=cache, use_cache=True
    )
    sampler = make_sampler(
        temp=request.talker_temperature,
        top_p=request.talker_top_p,
        top_k=request.talker_top_k,
    )
    codes: list[mx.array] = []
    tokens: list[int] = []
    delivered = 0
    eos = False
    suppressed = mx.array(
        [
            token
            for token in range(
                talker.config.text_config.vocab_size - 1024,
                talker.config.text_config.vocab_size,
            )
            if token != talker.config.codec_eos_token_id
        ]
    )
    for step in range(request.talker_max_tokens):
        check_cancel(should_cancel)
        scores = logits[:, -1, :]
        scores[:, suppressed] = -float("inf")
        if tokens and request.talker_repetition_penalty != 1:
            previous = mx.array(sorted(set(tokens)))
            values = scores[:, previous]
            scores[:, previous] = mx.where(
                values < 0,
                values * request.talker_repetition_penalty,
                values / request.talker_repetition_penalty,
            )
        token = sampler(scores).reshape(1, 1)
        value = token_id(token)
        if value == talker.config.codec_eos_token_id:
            eos = True
            break
        embeddings, residual = talker.prepare_inputs_for_generation(
            input_ids=token,
            past_hidden=hidden[:, -1:],
            trailing_text_hidden=conditioning.trailing,
            tts_pad_embed=conditioning.padding,
            generation_step=step,
            temperature=request.talker_temperature,
            top_p=0.8,
        )
        mx.eval(residual)
        codes.append(residual)
        tokens.append(value)
        if request.stream_audio and len(codes) - delivered >= request.chunk_frames:
            emit(waveform(codec, codes, delivered))
            delivered = len(codes)
        logits, hidden = talker(
            inputs_embeds=embeddings, past_key_values=cache, use_cache=True
        )
    check_cancel(should_cancel)
    if codes and delivered < len(codes):
        emit(waveform(codec, codes, delivered))
    return eos, len(codes)
