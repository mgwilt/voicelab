"""Local PersonaPlex loading, prompt conditioning, and frame inference."""

import platform
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from voicelab_models.personaplex.model import (
    FRAME_BYTES,
    FRAME_SECONDS,
    MAX_FRAMES,
    REVISION,
    SAMPLE_RATE,
    SOURCE_MODEL,
    Request,
)

if TYPE_CHECKING:
    from mlx_audio.codec.models.mimi.mimi import Mimi
    from mlx_audio.sts.models.moshi.lm import Lm
    from sentencepiece import SentencePieceProcessor

    from voicelab_inference.personaplex.mlx.scheduler import Scheduler


@dataclass(frozen=True)
class Event:
    kind: Literal["audio", "assistant_text_delta"]
    pcm: bytes = b""
    text: str = ""


@dataclass(frozen=True)
class Metrics:
    input_seconds: float
    output_seconds: float
    generation_seconds: float
    rtf: float | None
    first_pcm_seconds: float | None


class PersonaPlexMLX:
    def __init__(self) -> None:
        self._model: Lm | None = None
        self._codec: Mimi | None = None
        self._tokenizer: SentencePieceProcessor | None = None
        self._scheduler: Scheduler | None = None
        self._voices: str | None = None
        self._frames = 0
        self._output_samples = 0
        self._processing = 0.0
        self._first_pcm: float | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("PersonaPlex MLX requires Apple Silicon.")
        import mlx.core as mx
        import mlx.nn as nn
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError
        from mlx_audio.codec.models.mimi.mimi import Mimi, mimi_202407
        from mlx_audio.sts.models.moshi.lm import Lm
        from sentencepiece import SentencePieceProcessor

        from voicelab_inference.personaplex.mlx.network import configuration

        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal is unavailable.")
        try:
            hf_hub_download(SOURCE_MODEL, "config.json", revision=REVISION)
        except GatedRepoError as exc:
            raise RuntimeError(
                "Accept the PersonaPlex model license at "
                "https://huggingface.co/nvidia/personaplex-7b-v1, "
                "then run hf auth login."
            ) from exc
        model_path = hf_hub_download(
            SOURCE_MODEL, "model.safetensors", revision=REVISION
        )
        codec_path = hf_hub_download(
            SOURCE_MODEL,
            "tokenizer-e351c8d8-checkpoint125.safetensors",
            revision=REVISION,
        )
        tokenizer_path = hf_hub_download(
            SOURCE_MODEL, "tokenizer_spm_32k_3.model", revision=REVISION
        )
        voices = hf_hub_download(SOURCE_MODEL, "voices.tgz", revision=REVISION)
        config = configuration()
        model = Lm(config)
        model.set_dtype(mx.bfloat16)
        model.load_pytorch_weights(model_path, config, strict=True)
        nn.quantize(model, bits=8, group_size=64)
        codec = Mimi(mimi_202407(32))
        codec.load_pytorch_weights(codec_path, strict=True)
        mx.eval(model.parameters(), codec.parameters())
        mx.synchronize()
        self._model, self._codec = model, codec
        self._tokenizer = SentencePieceProcessor(tokenizer_path)
        self._voices = voices

    def start(self, request: Request) -> None:
        from voicelab_inference.personaplex.mlx.voice import (
            condition_voice,
            reference_wave,
        )

        waveform = reference_wave(request)
        self.load()

        from voicelab_inference.personaplex.mlx.network import PersonaNetwork
        from voicelab_inference.personaplex.mlx.scheduler import Scheduler

        model, codec, tokenizer, voices = (
            self._model,
            self._codec,
            self._tokenizer,
            self._voices,
        )
        if model is None or codec is None or tokenizer is None or voices is None:
            raise RuntimeError("PersonaPlex did not finish loading.")
        prompt = request.prompt.strip()
        wrapped = (
            prompt
            if prompt.startswith("<system>") and prompt.endswith("<system>")
            else f"<system> {prompt} <system>"
        )
        tokens: object = tokenizer.encode(wrapped)
        if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
            raise TypeError("Tokenizer must return integer text tokens.")
        text_tokens = tuple(int(token) for token in tokens)
        scheduler = Scheduler(PersonaNetwork(model, request))
        condition_voice(scheduler, codec, voices, request, waveform, text_tokens)
        codec.reset_state()
        codec.downsample.reset_state()
        codec.upsample.reset_state()
        self._scheduler = scheduler
        self._frames, self._output_samples = 0, 0
        self._processing, self._first_pcm = 0.0, None

    def push(self, pcm: bytes) -> tuple[Event, ...]:
        import mlx.core as mx

        if len(pcm) != FRAME_BYTES:
            raise ValueError("Send 1920 mono PCM16 samples per frame.")
        if self._frames >= MAX_FRAMES:
            raise ValueError("PersonaPlex input exceeds 120 seconds.")
        codec, scheduler, tokenizer = self._codec, self._scheduler, self._tokenizer
        if codec is None or scheduler is None or tokenizer is None:
            raise RuntimeError("No PersonaPlex session is active.")
        start = time.perf_counter()
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        codes = codec.encode_step(mx.array(samples)[None, None, :])
        if codes.ndim != 3 or codes.shape[0] != 1 or codes.shape[-1] != 1:
            raise ValueError("Mimi did not produce one input frame.")
        output = scheduler.step(codes[:, :8, :])
        events: list[Event] = []
        if output is not None:
            mx.eval(output)
            text_token = output[0, 0, 0].item()
            if not isinstance(text_token, int):
                raise TypeError("Expected an integer text token.")
            if text_token not in (0, 1, 2, 3):
                piece: object = tokenizer.id_to_piece(text_token)
                if not isinstance(piece, str):
                    raise TypeError("Tokenizer must return a text piece.")
                events.append(
                    Event("assistant_text_delta", text=piece.replace("▁", " "))
                )
            audio_codes = output[:, 1:9, :]
            if bool(mx.any((audio_codes < 0) | (audio_codes >= 2048)).item()):
                raise ValueError("Generated audio contains reserved codebook tokens.")
            audio = codec.decode_step(audio_codes)
            mx.eval(audio)
            waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
            if waveform.size == 0 or not np.isfinite(waveform).all():
                raise ValueError("Mimi produced empty or non-finite audio.")
            encoded = (np.clip(waveform, -1.0, 1.0) * 32767).astype("<i2").tobytes()
            self._output_samples += waveform.size
            events.append(Event("audio", pcm=encoded))
        mx.synchronize()
        self._processing += time.perf_counter() - start
        self._frames += 1
        if self._first_pcm is None and self._output_samples:
            self._first_pcm = self._processing
        return tuple(events)

    def metrics(self) -> Metrics:
        input_seconds = self._frames * FRAME_SECONDS
        return Metrics(
            input_seconds,
            self._output_samples / SAMPLE_RATE,
            self._processing,
            self._processing / input_seconds if input_seconds else None,
            self._first_pcm,
        )

    def close(self) -> None:
        if self._model is not None:
            import mlx.core as mx

            mx.synchronize()
        self._scheduler = None
        self._model = None
        self._codec = None
        self._tokenizer = None
        self._voices = None
