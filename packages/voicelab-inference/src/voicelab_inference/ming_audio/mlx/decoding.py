"""Cancellable greedy thinker decoding adapted from MLX-Audio; see NOTICE."""

from collections.abc import Callable

import mlx.core as mx
from mlx_audio.sts.models.ming_v3_audio.ming_v3_audio import Model
from transformers import PreTrainedTokenizerBase


def decode(
    model: Model,
    embeddings: mx.array,
    limit: int,
    should_cancel: Callable[[], bool],
) -> tuple[list[int], bool]:
    if embeddings.shape[1] + limit > model.config.text_config.max_position_embeddings:
        raise ValueError("Ming prompt and generation exceed the context window")
    tokenizer = model.tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError("Ming tokenizer did not load")
    cache = model.make_cache()
    hidden: mx.array | None = None
    for start in range(0, embeddings.shape[1], 256):
        if should_cancel():
            raise InterruptedError("Audio conversation cancelled")
        candidate = model.model(
            inputs_embeds=embeddings[:, start : start + 256], cache=cache
        )
        if not isinstance(candidate, mx.array):
            raise TypeError("Ming backbone must return hidden states")
        hidden = candidate
        mx.eval(hidden, [entry.state for entry in cache])
    if hidden is None:
        raise ValueError("Ming thinker received an empty prompt")
    configured = model.config.text_config.eos_token_id
    stops = set(configured if isinstance(configured, list) else [configured])
    stops.update(
        (
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|role_end|>"),
        )
    )
    tokens: list[int] = []
    for _ in range(limit):
        if should_cancel():
            raise InterruptedError("Audio conversation cancelled")
        logits = model.lm_head(hidden[:, -1]).astype(mx.float32)
        token = mx.argmax(logits, axis=-1)[0].item()
        if not isinstance(token, int):
            raise TypeError("Ming token index must be an integer")
        if token in stops:
            return tokens, True
        tokens.append(token)
        candidate = model.model(mx.array([[token]]), cache=cache)
        if not isinstance(candidate, mx.array):
            raise TypeError("Ming backbone must return hidden states")
        hidden = candidate
    return tokens, False
