"""Observe the resident LM's raw output before Kitten removes stop tokens."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from kittenml.kittentts2.tokens import TokenMap


@runtime_checkable
class Generate(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


@runtime_checkable
class LanguageModel(Protocol):
    generate: Generate


@dataclass(frozen=True)
class Stop:
    frames: int
    eos: bool


class GenerationObserver:
    def __init__(self, generate: Generate, tokens: TokenMap) -> None:
        self._generate = generate
        self._tokens = tokens
        self._pending: list[Stop] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        inputs = args[0] if args else kwargs.get("input_ids")
        if (
            not isinstance(inputs, torch.Tensor)
            or inputs.ndim != 2
            or inputs.shape[0] != 1
        ):
            raise TypeError("Kitten generation requires one prompt tensor")
        output = self._generate(*args, **kwargs)
        if (
            not isinstance(output, torch.Tensor)
            or output.ndim != 2
            or output.shape[0] != 1
        ):
            raise TypeError("Kitten returned an incompatible generated sequence")
        generated = output[0, inputs.shape[1] :]
        if generated.numel() == 0:
            raise ValueError("Kitten returned no generated tokens")
        last = int(generated[-1].item())
        frames = int(
            (
                (generated >= self._tokens.audio_id_base)
                & (generated < self._tokens.audio_id_end)
            )
            .sum()
            .item()
        )
        self._pending.append(
            Stop(frames, last in (self._tokens.speech_end_id, self._tokens.stop_id))
        )
        return output

    def take(self) -> Stop | None:
        if not self._pending:
            return None
        stop = Stop(
            sum(item.frames for item in self._pending),
            all(item.eos for item in self._pending),
        )
        self._pending.clear()
        return stop
