"""Frame-wise generation over the released Transformers Phonon model.

The causal codec re-decodes the prefix to preserve convolution history. This
baseline favors parity over speed; it emits PCM before generation finishes.
"""

import math
from collections.abc import Generator
from time import perf_counter

import numpy as np
import torch
from transformers import GradiumPhononTTSForConditionalGeneration, PocketTTSProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.gradium_phonon_tts.modeling_gradium_phonon_tts import (
    GradiumPhononTTSOutputWithPast,
)

from voicelab_inference.phonon.torch.contracts import Chunk
from voicelab_models.phonon.model import Request


def stream(
    model: GradiumPhononTTSForConditionalGeneration,
    processor: PocketTTSProcessor,
    features: BatchFeature,
    request: Request,
) -> Generator[Chunk]:
    tail = features.get("frames_after_eos")
    if not isinstance(tail, torch.Tensor) or tail.numel() != 1:
        raise ValueError("Phonon must expose its end-of-speech tail")
    inputs: dict[str, torch.Tensor] = {}
    for key in (
        "input_ids",
        "attention_mask",
        "audio_prompt_latents",
        "audio_prompt_latents_mask",
    ):
        value = features.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Phonon processor did not return {key}")
        inputs[key] = value
    output = model(**inputs, use_cache=True)
    latents: list[torch.Tensor] = []
    first_stop: int | None = None
    emitted = 0
    for step in range(request.max_frames):
        if not isinstance(output, GradiumPhononTTSOutputWithPast):
            raise TypeError("Phonon returned an unsupported model output")
        hidden = output.last_hidden_state
        scores = output.logits
        if not isinstance(hidden, torch.Tensor) or not isinstance(scores, torch.Tensor):
            raise ValueError("Phonon must return hidden state and EOS logits")
        condition = hidden[:, -1, :]
        noise = torch.randn(
            (1, model.config.latent_dim), device=condition.device, dtype=condition.dtype
        ) * math.sqrt(request.temperature)
        latent = model.sample_latent(condition, noise, num_flow_steps=1)[:, None, :]
        latents.append(latent)
        if first_stop is None and float(scores[0, -1, 0].item()) > 4:
            first_stop = step
        eos = first_stop is not None and step >= first_stop + int(tail.item())
        decoding = perf_counter()
        all_latents = model.denormalize_latents(torch.cat(latents, dim=1))
        decoded = processor.decode(
            all_latents,
            torch.ones((1, step + 1), device=condition.device, dtype=torch.long),
        )
        if not isinstance(decoded, list) or len(decoded) != 1:
            raise ValueError("Phonon decoder must return one waveform")
        waveform = decoded[0]
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 1:
            raise ValueError("Phonon decoder must return mono audio")
        audio = waveform[emitted:].detach().float().cpu().numpy()
        emitted = waveform.numel()
        yield Chunk(
            np.asarray(audio, dtype=np.float32),
            frames=1,
            eos=eos,
            codec_seconds=perf_counter() - decoding,
        )
        if eos:
            return
        output = model(
            input_ids=torch.full(
                (1, 1),
                model.config.audio_latent_token_id,
                device=condition.device,
                dtype=torch.long,
            ),
            audio_latents=latent,
            past_key_values=output.past_key_values,
            use_cache=True,
        )
