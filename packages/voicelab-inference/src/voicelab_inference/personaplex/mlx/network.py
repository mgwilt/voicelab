"""PersonaPlex architecture over pinned MLX-Audio Moshi primitives."""

from dataclasses import replace
from typing import Protocol

import mlx.core as mx
from mlx_audio.sts.models.moshi.lm import Lm, LmConfig, config_v0_1

from voicelab_models.personaplex.model import Request


def configuration() -> LmConfig:
    baseline = config_v0_1()
    return replace(
        baseline,
        transformer=replace(baseline.transformer, dim_feedforward=16896),
        depformer=replace(
            baseline.depformer,
            num_slices=16,
            transformer=replace(
                baseline.depformer.transformer, dim_feedforward=4224, context=16
            ),
        ),
    )


class Network(Protocol):
    def embed(self, tokens: mx.array) -> mx.array: ...
    def forward(self, embeddings: mx.array) -> tuple[mx.array, mx.array]: ...
    def depth(
        self,
        hidden: mx.array,
        text: mx.array,
        forced: mx.array,
        provided: mx.array,
    ) -> mx.array: ...
    def text(self, logits: mx.array) -> mx.array: ...
    def reset(self) -> None: ...


class PersonaNetwork:
    def __init__(self, model: Lm, request: Request | None = None) -> None:
        self.model = model
        self.request = request or Request()
        self._key = mx.random.key(self.request.seed)

    def reset(self) -> None:
        for cache in (*self.model.transformer_cache, *self.model.depformer_cache):
            cache.reset()

    def embed(self, tokens: mx.array) -> mx.array:
        embeddings = self.model.text_emb(tokens[:, 0, :])
        for index, embedding in enumerate(self.model.audio_embs):
            embeddings = embeddings + embedding(tokens[:, index + 1, :])
        return embeddings

    def forward(self, embeddings: mx.array) -> tuple[mx.array, mx.array]:
        hidden = self.model.transformer(embeddings, cache=self.model.transformer_cache)
        hidden = self.model.out_norm(hidden)
        if not isinstance(hidden, mx.array):
            raise TypeError("Temporal network must return an MLX array.")
        return hidden, self.model.text_linear(hidden)

    def text(self, logits: mx.array) -> mx.array:
        return self._sample(
            logits, self.request.text_temperature, self.request.text_top_k
        )

    def _sample(self, logits: mx.array, temperature: float, top_k: int) -> mx.array:
        if temperature == 0:
            return mx.argmax(logits, axis=-1).astype(mx.int32)
        self._key, sample_key = mx.random.split(self._key)
        scaled = logits / temperature
        excluded = mx.argpartition(-scaled, kth=top_k - 1, axis=-1)[..., top_k:]
        masked = mx.put_along_axis(
            scaled, excluded, mx.array(-float("inf"), scaled.dtype), axis=-1
        )
        return mx.random.categorical(masked, axis=-1, key=sample_key).astype(mx.int32)

    def depth(
        self,
        hidden: mx.array,
        text: mx.array,
        forced: mx.array,
        provided: mx.array,
    ) -> mx.array:
        # Teacher-forced tokens must condition each later codebook prediction.
        cache = self.model.depformer_cache
        for item in cache:
            item.reset()
        previous = text
        generated: list[mx.array] = []
        for index, layer in enumerate(self.model.depformer.slices):
            state = layer.linear_in(hidden) + layer.emb(previous)
            state = layer.transformer(state, cache=cache)
            sampled = self._sample(
                layer.linear_out(state),
                self.request.audio_temperature,
                self.request.audio_top_k,
            )
            previous = mx.where(
                provided[:, index : index + 1], forced[:, index : index + 1], sampled
            )
            generated.append(sampled)
        return mx.stack(generated, axis=1)
