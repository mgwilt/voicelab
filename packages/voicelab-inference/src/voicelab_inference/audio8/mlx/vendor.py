"""Convert pinned Realistic weights into the MLX-Audio ArkTTS architecture."""

import json
from collections.abc import Generator
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import snapshot_download
from mlx_audio.tts.models.arktts.arktts import Model, ModelConfig
from transformers import PreTrainedTokenizerFast

from voicelab_inference.audio import reference_pcm
from voicelab_inference.audio8.mlx.contracts import Chunk
from voicelab_models.audio8.model import MODEL_ID, MODEL_REVISION, SAMPLE_RATE, Request


def codec_weights(path: Path) -> dict[str, mx.array]:
    """Load only tensor data; reject unexpected checkpoint structures."""
    value: object = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("Audio8 codec checkpoint must be a tensor mapping")
    if "state_dict" in value:
        value = value["state_dict"]
    if not isinstance(value, dict):
        raise ValueError("Audio8 codec state_dict must be a tensor mapping")
    result: dict[str, mx.array] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("Audio8 codec contains a non-tensor entry")
        result[f"codec.{key}"] = mx.array(tensor.float().numpy())
    return result


class Vendor:
    def __init__(self) -> None:
        path = Path(
            snapshot_download(
                MODEL_ID,
                revision=MODEL_REVISION,
                allow_patterns=["*.json", "*.safetensors", "codec.pth"],
            )
        )
        config: object = json.loads((path / "config.json").read_text())
        if not isinstance(config, dict) or config.get("model_type") != "arktts":
            raise ValueError("Audio8 requires an ArkTTS checkpoint")
        model = Model(ModelConfig.from_dict(config))
        if model.sample_rate != SAMPLE_RATE:
            raise ValueError("Audio8 checkpoint sample rate changed")
        raw = mx.load(str(path / "model.safetensors"))
        if not isinstance(raw, dict):
            raise ValueError("Audio8 language model must contain named tensors")
        weights: dict[str, mx.array] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, mx.array):
                raise ValueError("Audio8 language model contains invalid weights")
            weights[key] = value.astype(mx.float32)
        weights.update(codec_weights(path / "codec.pth"))
        model.load_weights(list(model.sanitize(weights).items()), strict=True)
        model.model_path = path
        model._tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
        model.eval()
        mx.eval(model.parameters())
        self.model = model

    def generate(self, request: Request) -> Generator[Chunk]:
        prefix, suffix = self.model._prompt_segments(
            request.text, request.ref_text, request.ref_audio is not None
        )
        reference_codes = None
        reference_lengths = None
        if request.ref_audio is not None:
            waveform = reference_pcm(request.ref_audio, SAMPLE_RATE)
            encoded, lengths = self.model.encode_reference(mx.array(waveform))
            mx.eval(encoded, lengths)
            reference_codes = np.asarray(encoded)
            reference_lengths = np.asarray(lengths)
        prompt, _ = self.model.model._prepare_prompt(
            np.asarray([prefix]),
            np.asarray([suffix]),
            reference_codes,
            reference_lengths,
        )
        budget = min(
            request.max_frames, self.model.config.max_seq_len - prompt.shape[-1]
        )
        if budget < 1:
            raise ValueError("Audio8 text exceeds the context window")
        try:
            codes = self.model.model.generate_codes(
                [prefix],
                [suffix],
                reference_codes,
                reference_lengths,
                max_new_tokens=request.max_frames,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=request.do_sample,
                seed=request.seed,
            )
            mx.eval(codes)
            frames = codes.shape[-1]
            if frames == 0:
                raise RuntimeError("Audio8 generated no speech frames")
            start = perf_counter()
            waveform = self.model.codec.decode(codes)[0]
            mx.eval(waveform)
            yield Chunk(
                np.asarray(waveform, dtype=np.float32),
                frames=frames,
                # The vendor removes the sampled EOS frame from returned codes.
                eos=frames < budget,
                codec_seconds=perf_counter() - start,
            )
        finally:
            self.model.model._clear_generation_caches()
            mx.clear_cache()

    def close(self) -> None:
        self.model.model._clear_generation_caches()
        mx.clear_cache()
