"""Audio-only conversation preparation through the pinned native processor."""

from dataclasses import dataclass

import mlx.core as mx
from mlx_vlm.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from voicelab_inference.audio import reference_pcm
from voicelab_models.audio import ReferenceAudio
from voicelab_models.qwen_omni.model import INPUT_SAMPLE_RATE, Request


@dataclass(frozen=True)
class Inputs:
    ids: mx.array
    features: mx.array | None
    mask: mx.array | None


def prepare(processor: Qwen3OmniMoeProcessor, request: Request) -> Inputs:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": request.system_prompt}
    ]
    recordings: list[ReferenceAudio] = []

    def content(text: str, audio: ReferenceAudio | None) -> str:
        parts: list[str] = []
        if audio is not None:
            recordings.append(audio)
            parts.append("<|audio_start|><|audio_pad|><|audio_end|>")
        if text:
            parts.append(text)
        return "".join(parts)

    for turn in request.history:
        messages.append(
            {"role": turn.role, "content": content(turn.content, turn.audio)}
        )
    messages.append({"role": "user", "content": content(request.prompt, request.audio)})
    text: object = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    if not isinstance(text, str):
        raise TypeError("Qwen chat template must return text")
    audios = [reference_pcm(audio, INPUT_SAMPLE_RATE) for audio in recordings]
    raw = processor(
        text=text,
        audio=audios or None,
        sampling_rate=INPUT_SAMPLE_RATE,
        padding=True,
        return_attention_mask=True,
    )
    ids: object = raw.get("input_ids")
    features: object = raw.get("input_features")
    mask: object = raw.get("feature_attention_mask")
    if (
        not isinstance(ids, mx.array)
        or ids.ndim != 2
        or ids.shape[0] != 1
        or ids.shape[1] > 16000
    ):
        raise ValueError("Expected a single prompt of at most 16000 tokens")
    if features is not None and not isinstance(features, mx.array):
        raise TypeError("Qwen audio features must be MLX arrays")
    if mask is not None and not isinstance(mask, mx.array):
        raise TypeError("Qwen audio mask must be an MLX array")
    return Inputs(ids, features, mask)
