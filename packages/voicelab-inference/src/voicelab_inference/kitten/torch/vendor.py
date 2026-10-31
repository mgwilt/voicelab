"""Pinned KittenML 0.9.7 boundary with a pinned external speech decoder."""

import threading
from collections.abc import Generator
from contextlib import ExitStack, closing
from importlib import import_module
from typing import Literal, Protocol, TypedDict, runtime_checkable
from unittest.mock import patch

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from kittenml.kittentts2.tokens import TokenMap
from numpy.typing import NDArray

from voicelab_inference.audio import reference_file
from voicelab_inference.kitten.torch.completion import GenerationObserver, LanguageModel
from voicelab_inference.kitten.torch.contracts import Chunk
from voicelab_inference.kitten.torch.sampling import preserve_random_state
from voicelab_models.kitten.model import (
    CODEC_FILE,
    CODEC_ID,
    CODEC_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RATE,
    Request,
)

# The upstream codec loader has no revision argument. Serialize the scoped hook.
_LOAD_LOCK = threading.Lock()
type Device = Literal["cpu", "mps", "cuda"]
type Audio = NDArray[np.float32]


class Advanced(TypedDict):
    seed: int
    repetition_penalty: float
    repetition_window: int | None
    token_run_penalty: float
    token_run_grace: int
    chunk_chars: int | None
    chunk_min_chars: int | None
    use_emotion: bool | None


@runtime_checkable
class Model(Protocol):
    sample_rate: int
    available_voices: list[str]
    model: object
    token_map: object
    _reference_cache: object

    def generate_stream(
        self,
        text: str,
        *,
        voice: str | None,
        reference: str | None,
        reference_text: str | None,
        use_reference_prompt: bool | None,
        temperature: float | None,
        top_k: int | None,
        top_p: float | None,
        min_p: float | None,
        preset: str,
        max_new_tokens: int,
        normalize: bool,
        advanced: Advanced,
    ) -> Generator[object]: ...


@runtime_checkable
class Loader(Protocol):
    def __call__(
        self, path: str, *, backend: str, device: str, weights: str, decoder: str
    ) -> object: ...


def _codec_download(
    *, repo_id: str, filename: str, cache_dir: str | None = None
) -> str:
    if repo_id != CODEC_ID or filename != CODEC_FILE:
        raise ValueError("Kitten requested an unexpected external codec")
    return hf_hub_download(
        repo_id, filename, revision=CODEC_REVISION, cache_dir=cache_dir
    )


def validate_audio(value: object) -> Audio:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 1
        or value.size == 0
        or value.dtype.kind != "f"
    ):
        raise ValueError("Kitten returned an invalid mono waveform")
    if not np.isfinite(value).all():
        raise ValueError("Kitten returned non-finite audio")
    return np.asarray(value, dtype=np.float32)


class KittenModel:
    def __init__(self, device: Device) -> None:
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Kitten MPS requires an available Apple GPU")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Kitten CUDA requires an available NVIDIA GPU")
        self.device = device
        module = import_module("kittenml")
        loader: object = getattr(module, "KittenTTS", None)
        if not isinstance(loader, Loader):
            raise TypeError("KittenML 0.9.7 does not expose the expected loader")
        path = snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=[
                "config.json",
                "lm/*.json",
                "lm/*.txt",
                "lm/model-ternary.safetensors",
                "voices/*",
                "speaker/*",
            ],
        )
        vocoder = import_module("kittenml.kittentts2.vocoder")
        # Patch only the vendor module's bound download function, not HF globally.
        # The context restores the original function after success or failure.
        with _LOAD_LOCK, patch.object(vocoder, "hf_hub_download", _codec_download):
            model = loader(
                path,
                backend="torch",
                device=device,
                weights="packed",
                decoder="default",
            )
        if not isinstance(model, Model) or model.sample_rate != SAMPLE_RATE:
            raise TypeError("Kitten returned an incompatible model or sample rate")
        self._model: Model | None = model

    def chunks(self, request: Request) -> Generator[Chunk]:
        if self._model is None:
            raise RuntimeError("Kitten has been closed")
        if (
            request.voice is not None
            and request.voice not in self._model.available_voices
        ):
            raise ValueError(
                "The requested Kitten voice is not present in the checkpoint"
            )
        native = self._model.model
        tokens = self._model.token_map
        if not isinstance(native, LanguageModel) or not isinstance(tokens, TokenMap):
            raise TypeError("Kitten does not expose the expected native token boundary")
        observer = GenerationObserver(native.generate, tokens)
        with (
            preserve_random_state(),
            reference_file(request.ref_audio) as reference,
            patch.object(native, "generate", observer),
            ExitStack() as resources,
        ):
            if request.ref_audio is not None:
                cache = self._model._reference_cache
                if not isinstance(cache, dict):
                    raise TypeError("Kitten reference cache has an incompatible type")
                resources.callback(cache.clear)
            stream = self._model.generate_stream(
                request.text,
                voice=(request.voice or "Bella") if reference is None else None,
                reference=reference,
                reference_text=request.ref_text,
                use_reference_prompt=request.use_reference_prompt,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                min_p=request.min_p,
                preset=request.preset,
                max_new_tokens=request.max_tokens,
                normalize=request.normalize,
                advanced=Advanced(
                    seed=request.seed,
                    repetition_penalty=request.repetition_penalty,
                    repetition_window=request.repetition_window,
                    token_run_penalty=request.token_run_penalty,
                    token_run_grace=request.token_run_grace,
                    chunk_chars=request.chunk_chars,
                    chunk_min_chars=request.chunk_min_chars,
                    use_emotion=request.use_emotion,
                ),
            )
            with closing(stream):
                for value in stream:
                    audio = validate_audio(value)
                    if self.device == "mps":
                        torch.mps.synchronize()
                    elif self.device == "cuda":
                        torch.cuda.synchronize()
                    stop = observer.take()
                    if stop is None:
                        raise RuntimeError(
                            "Kitten emitted audio without native stop evidence"
                        )
                    yield Chunk(audio, stop.frames, stop.eos)
                pending = observer.take()
                if pending is not None:
                    yield Chunk(
                        np.zeros(0, dtype=np.float32), pending.frames, pending.eos
                    )

    def close(self) -> None:
        self._model = None
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
