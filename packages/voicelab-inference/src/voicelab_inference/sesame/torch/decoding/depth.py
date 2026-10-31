"""CSM frame-local depth decoding, adapted from csm-mps (see NOTICE)."""

import copy

import torch
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.generation.logits_process import TopPLogitsWarper
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.csm.modeling_csm import CsmDepthDecoderForCausalLM


def direct_depth_generate(
    decoder: CsmDepthDecoderForCausalLM,
    input_ids: Tensor,
    backbone_last_hidden_state: Tensor,
    **kwargs: object,
) -> Tensor:
    """Retain all codebooks and sampling; avoid generic generate per codebook."""
    if kwargs:
        raise ValueError(f"Unsupported direct depth options: {sorted(kwargs)}")
    settings = copy.copy(decoder.generation_config)
    if settings is None:
        raise ValueError("Depth generation configuration is missing")
    for name, value in settings._get_default_generation_params().items():
        if getattr(settings, name, None) is None:
            setattr(settings, name, value)
    if settings.repetition_penalty != 1 or settings.num_beams != 1:
        raise ValueError("Direct depth requires unpenalized single-beam decoding")
    sequence = [input_ids]
    cache: Cache | None = None
    token: Tensor | None = None
    top_p = TopPLogitsWarper(settings.top_p) if settings.top_p < 1 else None
    codebooks = decoder.model.config.num_codebooks
    if not isinstance(codebooks, int):
        raise TypeError("Invalid depth codebook count")
    for book in range(codebooks - 1):
        if book == 0:
            embeddings = decoder.model.embed_tokens(input_ids)
            embeddings[:, 0] = backbone_last_hidden_state
        else:
            if token is None:
                raise RuntimeError("Missing previous depth token")
            embeddings = decoder.model.embed_tokens(
                token + book * decoder.model.config.vocab_size
            )
        output = decoder.model(
            inputs_embeds=embeddings, past_key_values=cache, use_cache=True
        )
        if not isinstance(output, BaseModelOutputWithPast):
            raise TypeError("Unexpected depth model output")
        cache = output.past_key_values
        # Integer indexing avoids copying an entire head via tensor indexing.
        logits = torch.nn.functional.linear(
            output.last_hidden_state[:, -1], decoder.codebooks_head.weight[book].T
        ).float()
        if settings.do_sample:
            logits = logits / settings.temperature
            if settings.top_k:
                threshold = torch.topk(logits, min(settings.top_k, logits.shape[-1]))[
                    0
                ][:, -1:]
                logits = logits.masked_fill(logits < threshold, -float("inf"))
            if top_p is not None:
                logits = top_p(input_ids, logits)
            token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        else:
            token = logits.argmax(dim=-1, keepdim=True)
        sequence.append(token)
    return torch.cat(sequence, dim=1)
