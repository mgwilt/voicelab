"""Request-owned causal history, adapted from csm-mps (see NOTICE)."""

from collections.abc import Callable
from typing import Protocol

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.models.mimi.modeling_mimi import (
    MimiConv1d,
    MimiConv1dPaddingCache,
    MimiConvTranspose1d,
    MimiModel,
)

from voicelab_inference.sesame.torch.audio import Audio, tensor_audio


class ConvForward(Protocol):
    def __call__(
        self, hidden_states: Tensor, padding_cache: MimiConv1dPaddingCache | None = None
    ) -> Tensor: ...


class TransposeForward(Protocol):
    def __call__(self, hidden_states: Tensor) -> Tensor: ...


class ConvolutionHistory:
    """Install temporary forward closures on an exclusively owned decoder."""

    def __init__(self, decoder: nn.Module) -> None:
        self.originals: list[tuple[nn.Module, Callable[..., Tensor]]] = []
        try:
            for module in decoder.modules():
                if isinstance(module, MimiConv1d):
                    if not module.causal or module.conv.stride != (1,):
                        raise ValueError(
                            "Streaming requires causal stride-one convolution"
                        )
                    self.originals.append((module, module.forward))
                    # Vendor annotation includes unbound self; closure is bound.
                    module.forward = self._conv(module)  # ty: ignore[invalid-assignment]
                elif isinstance(module, MimiConvTranspose1d):
                    if not module.causal or module.padding_left != 0:
                        raise ValueError(
                            "Streaming requires causal right-trimmed upsampling"
                        )
                    self.originals.append((module, module.forward))
                    # Reference uses the same instance-level forward adaptation.
                    module.forward = self._transpose(module)  # ty: ignore[invalid-assignment]
        except Exception:
            self.close()
            raise

    @staticmethod
    def _conv(module: MimiConv1d) -> ConvForward:
        previous: Tensor | None = None
        original = module.forward
        padding = int(module.padding_total)

        def forward(
            hidden_states: Tensor, padding_cache: MimiConv1dPaddingCache | None = None
        ) -> Tensor:
            nonlocal previous
            values = hidden_states
            if padding_cache is not None:
                raise ValueError("External convolution cache is unsupported")
            if previous is None:
                result = original(values)
                history = module._pad1d(values, (padding, 0), mode=module.pad_mode)
            else:
                history = torch.cat((previous, values), dim=-1)
                result = module.conv(history)
            if padding:
                previous = history[..., -padding:].clone()
            if not isinstance(result, Tensor):
                raise TypeError("Convolution returned no tensor")
            return result

        return forward

    @staticmethod
    def _transpose(module: MimiConvTranspose1d) -> TransposeForward:
        previous: Tensor | None = None
        padding = module.padding_right

        def forward(hidden_states: Tensor) -> Tensor:
            nonlocal previous
            values = hidden_states
            result = module.conv(values)
            if previous is not None:
                result[..., :padding] += previous
            end = result.shape[-1] - padding
            if padding:
                previous = result[..., end:].clone()
                if module.conv.bias is not None:
                    previous -= module.conv.bias[None, :, None]
            if not isinstance(result, Tensor):
                raise TypeError("Transpose convolution returned no tensor")
            return result[..., :end]

        return forward

    def close(self) -> None:
        for module, forward in reversed(self.originals):
            module.forward = forward
        self.originals.clear()


class StatefulMimi:
    def __init__(self, codec: MimiModel) -> None:
        if codec.decoder is None:
            raise ValueError("Mimi decoder is missing")
        self.codec = codec
        self.history = ConvolutionHistory(codec.decoder)
        self.cache: Cache | None = None

    @torch.inference_mode()
    def decode(self, codes: Tensor) -> Audio:
        output = self.codec.decode(
            codes.transpose(1, 2), decoder_past_key_values=self.cache, use_cache=True
        )
        self.cache = output.decoder_past_key_values
        if not isinstance(output.audio_values, Tensor):
            raise TypeError("Mimi returned no audio tensor")
        return tensor_audio(output.audio_values[0, 0])

    def close(self) -> None:
        self.history.close()
        self.cache = None
