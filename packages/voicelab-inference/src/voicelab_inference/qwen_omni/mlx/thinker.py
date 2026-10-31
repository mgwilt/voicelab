"""Cancellable Thinker decoding retaining the Talker's accepted hidden layer."""

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
from mlx_vlm.models import cache as kv_cache
from mlx_vlm.models.base import LanguageModelOutput
from mlx_vlm.models.qwen3_omni_moe.qwen3_omni_moe import Model
from mlx_vlm.sample_utils import make_sampler
from transformers import PreTrainedTokenizerBase

from voicelab_inference.qwen_omni.mlx.inputs import Inputs
from voicelab_inference.qwen_omni.mlx.sampling import check_cancel, token_id
from voicelab_models.qwen_omni.model import Request


@dataclass(frozen=True)
class Thought:
    sequences: mx.array
    hidden: mx.array
    embeddings: mx.array
    text: str
    tokens: int
    eos: bool


def hidden(output: LanguageModelOutput) -> mx.array:
    value: object = output.hidden_states
    if not isinstance(value, mx.array):
        raise TypeError("Expected the captured Thinker hidden-layer array")
    return value


def decode(
    model: Model,
    tokenizer: PreTrainedTokenizerBase,
    inputs: Inputs,
    request: Request,
    emit: Callable[[str], None],
    should_cancel: Callable[[], bool],
) -> Thought:
    check_cancel(should_cancel)
    language = model.thinker.language_model
    features = model.thinker.get_input_embeddings(
        inputs.ids, input_features=inputs.features, feature_attention_mask=inputs.mask
    )
    embeddings = features.inputs_embeds
    cache = kv_cache.make_prompt_cache(language)
    layer = model.config.talker_config.accept_hidden_layer + 1
    output = language(
        inputs.ids,
        inputs_embeds=embeddings,
        cache=cache,
        position_ids=features.position_ids,
        rope_deltas=features.rope_deltas,
        output_hidden_state_idx=layer,
    )
    states = [hidden(output)]
    tokens: list[int] = []
    token_arrays: list[mx.array] = []
    sampler = make_sampler(
        temp=request.temperature, top_p=request.top_p, top_k=request.top_k
    )
    text = ""
    eos = False
    for _ in range(request.max_tokens):
        check_cancel(should_cancel)
        token = sampler(output.logits[:, -1, :]).reshape(1, 1)
        next_id = token_id(token)
        tokens.append(next_id)
        token_arrays.append(token.astype(inputs.ids.dtype))
        output = language(
            token,
            cache=cache,
            rope_deltas=features.rope_deltas,
            output_hidden_state_idx=layer,
        )
        states.append(hidden(output))
        updated = decode_text(tokenizer, tokens)
        if updated.startswith(text) and not updated.endswith("\ufffd"):
            emit(updated[len(text) :])
            text = updated
        if next_id == model.config.eos_token_id:
            eos = True
            break
    generated = mx.concatenate(token_arrays, axis=1)
    embeddings = mx.concatenate(
        [embeddings, language.model.embed_tokens(generated)], axis=1
    )
    final = decode_text(tokenizer, tokens)
    if final.startswith(text):
        emit(final[len(text) :])
    return Thought(
        mx.concatenate([inputs.ids, generated], axis=1),
        mx.concatenate(states, axis=1),
        embeddings,
        final,
        len(tokens),
        eos,
    )


def decode_text(tokenizer: PreTrainedTokenizerBase, tokens: list[int]) -> str:
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    if not isinstance(text, str):
        raise TypeError("Tokenizer must decode one text sequence")
    return text
