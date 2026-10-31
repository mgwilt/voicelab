"""Breeze sampling order adapted from breeze-tts-mps; see NOTICE."""

import mlx.core as mx

from voicelab_models.breeze.model import CODEC_VOCAB_SIZE


def guidance(conditional: mx.array, unconditional: mx.array, scale: float) -> mx.array:
    conditional = conditional.astype(mx.float32)
    unconditional = unconditional.astype(mx.float32)
    return unconditional + scale * (conditional - unconditional)


def backbone_scores(
    logits: mx.array, *, temperature: float = 0.9, top_k: int = 50, top_p: float = 1.0
) -> mx.array:
    """Apply HF-style warpers before reserved-ID masking; preserve separate EOS."""
    scores = logits / temperature
    if top_k:
        threshold = mx.sort(scores, axis=-1)[..., -min(top_k, scores.shape[-1]), None]
        scores = mx.where(scores < threshold, -mx.inf, scores)
    if top_p < 1:
        order = mx.argsort(scores, axis=-1)
        sorted_scores = mx.take_along_axis(scores, order, axis=-1)
        remove = mx.cumsum(mx.softmax(sorted_scores, axis=-1), axis=-1) <= 1 - top_p
        remove = mx.concatenate([remove[..., :-1], mx.zeros_like(remove[..., -1:])], -1)
        remove = mx.take_along_axis(remove, mx.argsort(order, axis=-1), axis=-1)
        scores = mx.where(remove, -mx.inf, scores)
    index = mx.arange(scores.shape[-1])
    valid = (index < CODEC_VOCAB_SIZE) | (index == scores.shape[-1] - 1)
    return mx.where(valid, scores, -mx.inf)


def depth_scores(
    logits: mx.array, *, temperature: float = 0.9, top_k: int = 50, top_p: float = 1.0
) -> mx.array:
    """Mask controls, then top-k renormalization and crossing-inclusive nucleus."""
    valid = mx.arange(logits.shape[-1]) < CODEC_VOCAB_SIZE
    probs = mx.softmax(mx.where(valid, logits / temperature, -mx.inf), axis=-1)
    order = mx.argsort(-probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, order, axis=-1)
    if top_k:
        sorted_probs = mx.where(mx.arange(probs.shape[-1]) < top_k, sorted_probs, 0)
        sorted_probs /= mx.sum(sorted_probs, axis=-1, keepdims=True)
    if top_p < 1:
        remove = mx.cumsum(sorted_probs, axis=-1) > top_p
        remove = mx.concatenate([mx.zeros_like(remove[..., :1]), remove[..., :-1]], -1)
        sorted_probs = mx.where(remove, 0, sorted_probs)
        sorted_probs /= mx.sum(sorted_probs, axis=-1, keepdims=True)
    return mx.log(mx.take_along_axis(sorted_probs, mx.argsort(order, axis=-1), -1))


class Sampler:
    """Request-local PRNG state, independent of other lab experiments."""

    def __init__(self, seed: int) -> None:
        self._key = mx.random.key(seed)

    def sample(self, scores: mx.array) -> mx.array:
        invalid = mx.any(mx.isnan(scores) | mx.isposinf(scores))
        if bool(invalid.item()) or not bool(
            mx.all(mx.any(mx.isfinite(scores), -1)).item()
        ):
            raise ValueError("Invalid or fully filtered sampling distribution")
        self._key, sample_key = mx.random.split(self._key)
        return mx.random.categorical(scores, key=sample_key).astype(mx.int32)
