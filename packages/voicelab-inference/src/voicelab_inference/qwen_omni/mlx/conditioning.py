"""Project user audio and generated text into the native Talker timeline."""

from dataclasses import dataclass
from itertools import pairwise

import mlx.core as mx
import numpy as np
from mlx_vlm.models.qwen3_omni_moe.qwen3_omni_moe import Model

from voicelab_inference.qwen_omni.mlx.sampling import token_id
from voicelab_inference.qwen_omni.mlx.thinker import Thought
from voicelab_models.qwen_omni.model import Speaker


@dataclass(frozen=True)
class Conditioning:
    embeddings: mx.array
    trailing: mx.array
    padding: mx.array
    input_ids: mx.array


def prepare(
    model: Model, original: mx.array, thought: Thought, speaker: Speaker
) -> Conditioning:
    talker = model.talker
    if talker is None:
        raise ValueError("Checkpoint has no Talker")
    config = model.config
    starts = np.flatnonzero(
        np.asarray(original[0]) == config.im_start_token_id
    ).tolist()
    starts.append(thought.sequences.shape[1])
    multimodal = thought.sequences == config.thinker_config.audio_token_id
    specials = mx.array(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
        dtype=original.dtype,
    )
    projected = talker.text_projection(
        model.thinker.language_model.model.embed_tokens(specials)
    )
    bos, eos, pad = projected[:, 0:1], projected[:, 1:2], projected[:, 2:3]
    embeddings: list[mx.array] = []
    ids: list[mx.array] = []
    trailing: mx.array | None = None
    for index, (start, end) in enumerate(pairwise(starts)):
        role = token_id(original[0, start + 1])
        if role == config.user_token_id:
            embeddings.append(
                model._get_talker_user_parts(
                    start, end, multimodal, thought.hidden, thought.embeddings
                )
            )
            ids.append(thought.sequences[:, start:end])
        elif role == config.assistant_token_id and index == len(starts) - 2:
            assistant, assistant_ids, raw_trailing = model._get_talker_assistant_parts(
                start,
                end,
                config.talker_config.speaker_id[speaker.lower()],
                thought.embeddings,
                pad,
                bos,
                eos,
            )
            if not isinstance(raw_trailing, mx.array):
                raise TypeError("Talker text conditioning must be an array")
            trailing = raw_trailing
            embeddings.append(assistant)
            ids.append(assistant_ids)
    if not embeddings or trailing is None:
        raise ValueError("Qwen did not prepare an assistant speech segment")
    return Conditioning(
        mx.concatenate(embeddings, axis=1), trailing, pad, mx.concatenate(ids, axis=1)
    )
