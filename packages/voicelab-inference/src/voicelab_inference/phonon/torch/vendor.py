"""Native Transformers Phonon with a separately pinned audio decoder."""

from collections.abc import Generator
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import (
    AutoTokenizer,
    GradiumPhononTTSForConditionalGeneration,
    PocketTTSAudioTokenizerModel,
    PocketTTSFeatureExtractor,
    PocketTTSProcessor,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.pocket_tts.generation_pocket_tts import PocketTTSGenerateOutput

from voicelab_inference.phonon.torch.completion import reached_eos
from voicelab_inference.phonon.torch.contracts import Chunk
from voicelab_inference.phonon.torch.streaming import stream
from voicelab_models.phonon.model import (
    CODEC_ID,
    CODEC_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RATE,
    Request,
)
from voicelab_models.phonon.voices import VOICE_REPOSITORY, VOICE_REVISION, VOICES


class Vendor:
    def __init__(self, device: Literal["cpu", "mps", "cuda"]) -> None:
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Phonon MPS requires an available Metal device")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Phonon CUDA requires an available NVIDIA device")
        self.device = device
        path = Path(
            snapshot_download(
                MODEL_ID,
                revision=MODEL_REVISION,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.jinja",
                    "tokenizer.model",
                ],
            )
        )
        self.model = GradiumPhononTTSForConditionalGeneration.from_pretrained(
            path, dtype=torch.float32
        )
        self.codec = PocketTTSAudioTokenizerModel.from_pretrained(
            CODEC_ID, revision=CODEC_REVISION, dtype=torch.float32
        )
        torch.nn.Module.to(self.model, device)
        torch.nn.Module.to(self.codec, device)
        self.model.eval()
        self.codec.eval()
        # Construct explicitly so the processor cannot fetch an unpinned codec.
        self.processor = PocketTTSProcessor(
            feature_extractor=PocketTTSFeatureExtractor(
                sampling_rate=SAMPLE_RATE,
                frame_size=1920,
            ),
            tokenizer=AutoTokenizer.from_pretrained(path),
            audio_tokenizer=self.codec,
            chat_template=(path / "chat_template.jinja").read_text(),
            frames_after_eos=1,
            frames_after_eos_short_text=3,
            fade_in_duration=0.0,
        )
        voice = load_file(str(path / "default-voice.safetensors"))["speaker_wavs"]
        if voice.ndim != 3 or voice.shape[0] != 1:
            raise ValueError("Phonon default voice has an unsupported shape")
        self.voice = voice[0].T.float()
        self.voice_path = Path(
            snapshot_download(
                VOICE_REPOSITORY,
                revision=VOICE_REVISION,
                allow_patterns=list(VOICES.values()),
            )
        )

    def generate(self, request: Request) -> Generator[Chunk]:
        voice = self.voice
        if request.voice != "default":
            tensor = load_file(str(self.voice_path / VOICES[request.voice]))[
                "speaker_wavs"
            ]
            if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[1] != 512:
                raise ValueError("Phonon voice must contain [1, 512, frames] latents")
            voice = tensor[0].T.float()
        # Transformers 5.22 annotates text-only messages here; its documented
        # PocketTTS example requires multimodal content (audio plus text).
        inputs = self.processor.apply_chat_template(
            [  # ty: ignore[invalid-argument-type]
                {
                    "role": "user",
                    "content": [
                        {"type": "audio"},
                        {"type": "text", "text": request.text},
                    ],
                }
            ],
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            processor_kwargs={"audio_prompt_latents": [voice]},
        )
        if not isinstance(inputs, BatchFeature):
            raise TypeError("Phonon processor did not return model features")
        inputs = inputs.to(self.device)
        devices = [] if self.device == "cpu" else [0]
        with (
            torch.inference_mode(),
            torch.random.fork_rng(devices=devices, device_type=self.device),
        ):
            torch.random.default_generator.manual_seed(request.seed)
            if self.device == "mps":
                torch.mps.manual_seed(request.seed)
            elif self.device == "cuda":
                torch.cuda.manual_seed(request.seed)
            if request.streaming:
                chunks = stream(self.model, self.processor, inputs, request)
                try:
                    yield from chunks
                finally:
                    chunks.close()
                return
            output = self.model.generate(
                **inputs,
                max_new_tokens=request.max_frames,
                return_dict_in_generate=True,
                output_scores=True,
                eos_threshold=4.0,
                min_frames_before_eos=0,
                do_sample=request.temperature > 0,
                temperature=request.temperature if request.temperature > 0 else 1.0,
            )
            if not isinstance(output, PocketTTSGenerateOutput):
                raise TypeError("Phonon returned an unsupported generation output")
            latents, mask = output.audio_latents, output.audio_latents_mask
            if not isinstance(latents, torch.Tensor) or not isinstance(
                mask, torch.Tensor
            ):
                raise ValueError(
                    "Phonon did not return audio latents and a validity mask"
                )
            if latents.shape[0] != 1 or mask.shape[0] != 1:
                raise ValueError("Phonon supports one utterance per request")
            frames = int(mask.sum().item())
            scores = output.scores
            tail = inputs.get("frames_after_eos")
            if (
                scores is None
                or not isinstance(tail, torch.Tensor)
                or tail.numel() != 1
            ):
                raise ValueError("Phonon did not expose completion metadata")
            stop_scores = tuple(float(score[0].item()) for score in scores)
            eos = reached_eos(stop_scores, int(tail.item()))
            start = perf_counter()
            decoded = self.processor.decode(latents, mask)
            if not isinstance(decoded, list) or len(decoded) != 1:
                raise ValueError("Phonon decoder must return one waveform")
            waveform = decoded[0]
            if not isinstance(waveform, torch.Tensor) or waveform.ndim != 1:
                raise ValueError("Phonon decoder must return mono audio")
            audio = waveform.detach().float().cpu().numpy()
            yield Chunk(
                np.asarray(audio, dtype=np.float32),
                frames=frames,
                eos=eos,
                codec_seconds=perf_counter() - start,
            )

    def close(self) -> None:
        torch.nn.Module.to(self.model, "cpu")
        torch.nn.Module.to(self.codec, "cpu")
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
