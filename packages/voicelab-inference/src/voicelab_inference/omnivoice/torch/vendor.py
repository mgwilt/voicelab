"""Validated boundary for OmniVoice 0.2.1's public inference API."""

from importlib import import_module
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import torch
from huggingface_hub import snapshot_download
from numpy.typing import NDArray

from voicelab_inference.audio import reference_file
from voicelab_inference.omnivoice.torch.sampling import preserve_random_state
from voicelab_models.omnivoice.model import (
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RATE,
    Request,
)

type Device = Literal["cpu", "mps", "cuda"]
type Audio = NDArray[np.float32]


@runtime_checkable
class Model(Protocol):
    sampling_rate: int

    def generate(
        self,
        *,
        text: str,
        language: str | None,
        instruct: str | None,
        num_step: int,
        speed: float,
        ref_audio: str | None,
        ref_text: str | None,
        duration: float | None,
        normalize_text: bool,
        denoise: bool,
        preprocess_prompt: bool,
        postprocess_output: bool,
        guidance_scale: float,
        t_shift: float,
        layer_penalty_factor: float,
        position_temperature: float,
        class_temperature: float,
        audio_chunk_duration: float,
        audio_chunk_threshold: float,
        pad_duration: float,
        fade_duration: float,
    ) -> object: ...


@runtime_checkable
class Loader(Protocol):
    def from_pretrained(
        self, path: str, *, device_map: str, dtype: torch.dtype
    ) -> object: ...


def validate_audio(value: object) -> Audio:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("OmniVoice must return exactly one waveform")
    audio: object = value[0]
    if (
        not isinstance(audio, np.ndarray)
        or audio.ndim != 1
        or audio.size == 0
        or audio.dtype.kind != "f"
    ):
        raise ValueError("OmniVoice returned an invalid mono waveform")
    if not np.isfinite(audio).all():
        raise ValueError("OmniVoice returned non-finite audio")
    return np.asarray(audio, dtype=np.float32)


class OmniVoiceModel:
    def __init__(self, device: Device) -> None:
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("OmniVoice MPS requires an available Apple GPU")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("OmniVoice CUDA requires an available NVIDIA GPU")
        self.device = device
        module = import_module("omnivoice")
        loader: object = getattr(module, "OmniVoice", None)
        if not isinstance(loader, Loader):
            raise TypeError("OmniVoice 0.2.1 does not expose the expected loader")
        path = snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=["*.json", "*.safetensors", "audio_tokenizer/*"],
        )
        model = loader.from_pretrained(
            path,
            device_map=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        if not isinstance(model, Model) or model.sampling_rate != SAMPLE_RATE:
            raise TypeError("OmniVoice returned an incompatible model or sample rate")
        self._model: Model | None = model

    def generate(self, request: Request) -> Audio:
        if self._model is None:
            raise RuntimeError("OmniVoice has been closed")
        with preserve_random_state(), reference_file(request.ref_audio) as reference:
            torch.manual_seed(request.seed)
            result = self._model.generate(
                text=request.text,
                language=request.language,
                instruct=request.instruction,
                num_step=request.steps,
                speed=request.speed,
                ref_audio=reference,
                ref_text=request.ref_text,
                duration=request.duration,
                normalize_text=request.normalize,
                denoise=request.denoise,
                preprocess_prompt=request.preprocess_prompt,
                postprocess_output=request.postprocess_output,
                guidance_scale=request.guidance_scale,
                t_shift=request.t_shift,
                layer_penalty_factor=request.layer_penalty_factor,
                position_temperature=request.position_temperature,
                class_temperature=request.class_temperature,
                audio_chunk_duration=request.audio_chunk_duration,
                audio_chunk_threshold=request.audio_chunk_threshold,
                pad_duration=request.pad_duration,
                fade_duration=request.fade_duration,
            )
            audio = validate_audio(result)
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()
        return audio

    def close(self) -> None:
        self._model = None
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
