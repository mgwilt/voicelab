"""Delayed streams and teacher-forced voice/text prompts for PersonaPlex."""

import mlx.core as mx

from voicelab_inference.personaplex.mlx.network import Network

DELAYS = (0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1)
SILENCE = (948, 243, 1178, 546, 1736, 1030, 1978, 2008)
SINE = (430, 1268, 381, 1611, 1095, 1495, 56, 472)


class Scheduler:
    """One stream owns four delayed-token slots and its network caches."""

    def __init__(self, network: Network) -> None:
        self.network = network
        self.offset = 0
        self.cache = mx.full((1, 17, 4), -2, dtype=mx.int32)
        self.provided = mx.zeros((1, 17, 4), dtype=mx.bool_)
        self.initial = mx.array([32000, *([2048] * 16)], dtype=mx.int32)[None, :]
        network.reset()

    def _prepare(
        self,
        user: mx.array,
        agent: mx.array | None,
        text: int | None,
    ) -> tuple[mx.array, mx.array, mx.array, int, int] | None:
        if self.offset >= 4096:
            raise ValueError("PersonaPlex context limit reached.")
        if user.shape != (1, 8, 1):
            raise ValueError("Expected eight input codebooks for one frame.")
        if agent is not None and agent.shape != (1, 8, 1):
            raise ValueError("Expected eight prompt codebooks for one frame.")
        for index in range(8):
            stream = 9 + index
            position = (self.offset + DELAYS[stream]) % 4
            self.cache[:, stream, position : position + 1] = user[:, index, :]
            self.provided[:, stream, position] = True
        if agent is not None:
            for index in range(8):
                stream = 1 + index
                position = (self.offset + DELAYS[stream]) % 4
                self.cache[:, stream, position : position + 1] = agent[:, index, :]
                self.provided[:, stream, position] = True
        if text is not None:
            position = self.offset % 4
            self.cache[:, 0, position] = text
            self.provided[:, 0, position] = True
        for stream, delay in enumerate(DELAYS):
            if self.offset <= delay:
                self.cache[:, stream, self.offset % 4] = self.initial[:, stream]
                self.provided[:, stream, self.offset % 4] = True
        if self.offset == 0:
            self.cache[:, :, 0] = self.initial
            self.offset = 1
            return None
        previous, target = (self.offset - 1) % 4, self.offset % 4
        inputs = self.cache[:, :, previous : previous + 1]
        if bool(mx.any(inputs == -2).item()):
            raise ValueError("A delayed stream token has not been generated.")
        return (
            inputs,
            self.provided[:, :, target : target + 1],
            self.cache[:, :, target : target + 1],
            previous,
            target,
        )

    def step(
        self,
        user: mx.array,
        agent: mx.array | None = None,
        text: int | None = None,
        embeddings: mx.array | None = None,
    ) -> mx.array | None:
        prepared = self._prepare(user, agent, text)
        if prepared is None:
            return None
        inputs, provided, target, previous_position, target_position = prepared
        hidden, logits = self.network.forward(
            self.network.embed(inputs) if embeddings is None else embeddings
        )
        sampled_text = self.network.text(logits)[:, 0]
        next_text = mx.where(provided[:, 0, 0], target[:, 0, 0], sampled_text)
        sampled_audio = self.network.depth(
            hidden, next_text[:, None], target[:, 1:, 0], provided[:, 1:, 0]
        )
        self.provided[:, :, previous_position] = False
        self.cache[:, 0, target_position] = mx.where(
            self.provided[:, 0, target_position],
            self.cache[:, 0, target_position],
            sampled_text,
        )
        self.cache[:, 1:, target_position] = mx.where(
            self.provided[:, 1:, target_position],
            self.cache[:, 1:, target_position],
            sampled_audio[:, :, 0],
        )
        output: mx.array | None = None
        if self.offset > 1:
            output = mx.concatenate(
                [
                    self.cache[
                        :,
                        stream : stream + 1,
                        (self.offset - 1 + delay) % 4 : (self.offset - 1 + delay) % 4
                        + 1,
                    ]
                    for stream, delay in enumerate(DELAYS[:9])
                ],
                axis=1,
            )
        self.offset += 1
        mx.eval(self.cache, self.provided)
        return output

    def prompt(
        self,
        embeddings: mx.array,
        saved_cache: mx.array,
        text: tuple[int, ...],
    ) -> None:
        if embeddings.ndim != 4 or embeddings.shape[1:] != (1, 1, 4096):
            raise ValueError("Voice embeddings must have shape [frames, 1, 1, 4096].")
        if not 1 <= embeddings.shape[0] <= 1500 or saved_cache.shape != (1, 17, 4):
            raise ValueError("Voice prompt exceeds its shape or duration limit.")
        if len(text) > 1024:
            raise ValueError("PersonaPlex instructions exceed 1024 tokens.")
        dummy = mx.full((1, 8, 1), 2048, dtype=mx.int32)
        for embedding in embeddings:
            # First step initializes delay slots without executing the network.
            if self.offset == 0:
                self.step(dummy, dummy, 3)
            self.step(dummy, dummy, 3, embeddings=embedding)
        self.cache = saved_cache.astype(mx.int32)
        self.instructions(text)

    def instructions(self, text: tuple[int, ...]) -> None:
        if len(text) > 1024:
            raise ValueError("PersonaPlex instructions exceed 1024 tokens.")
        sine = mx.array(SINE, dtype=mx.int32).reshape(1, 8, 1)
        silence = mx.array(SILENCE, dtype=mx.int32).reshape(1, 8, 1)
        for token in (*([3] * 6), *text, *([3] * 6)):
            self.step(sine, silence, token)
        mx.synchronize()
