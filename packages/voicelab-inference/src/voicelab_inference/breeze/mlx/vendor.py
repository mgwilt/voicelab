"""Pinned MLX-Audio architecture boundary; lab owns sampling and stream policy."""

from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_audio.tts.models.breeze_tts.breeze_tts import Model
from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import Qwen3TTSSpeechTokenizer
from mlx_audio.tts.utils import load_model

from voicelab_inference.audio import reference_pcm
from voicelab_inference.breeze.mlx.sampling import (
    Sampler,
    backbone_scores,
    depth_scores,
    guidance,
)
from voicelab_models.breeze.model import (
    CODEBOOKS,
    CODEC_VOCAB_SIZE,
    EOS_TOKEN,
    MLX_MODEL_ID,
    MLX_REVISION,
    SAMPLE_RATE,
)
from voicelab_models.breeze.synthesis import SynthesisRequest


class BreezeModel:
    """Concrete, validated vendor model and request-local decoder caches."""

    def __init__(self) -> None:
        path = snapshot_download(MLX_MODEL_ID, revision=MLX_REVISION)
        model = load_model(Path(path), strict=True)
        if not isinstance(model, Model):
            raise TypeError("Checkpoint did not load a Breeze model")
        if (
            model.num_codebooks != CODEBOOKS
            or model.vocab_size != EOS_TOKEN
            or model.config.codec_vocab_size != CODEC_VOCAB_SIZE
            or model.sample_rate != SAMPLE_RATE
        ):
            raise ValueError("Unsupported Breeze checkpoint layout")
        codec = model.audio_tokenizer
        if not isinstance(codec, Qwen3TTSSpeechTokenizer):
            raise TypeError("Breeze requires the Qwen3-TTS speech tokenizer")
        self._model = model
        self._decoder = codec.decoder

    def reset_codec(self) -> None:
        self._decoder.reset_streaming_state()

    def decode(self, frames: list[mx.array]) -> mx.array:
        codes = mx.stack(frames, axis=0)[None, :, :]
        audio = self._decoder.streaming_step(mx.transpose(codes, (0, 2, 1)))
        if (
            not isinstance(audio, mx.array)
            or audio.ndim != 3
            or audio.shape[:2] != (1, 1)
        ):
            raise ValueError("Codec must return mono [1, 1, samples] audio")
        return audio[0, 0].astype(mx.float32)

    def frames(self, request: SynthesisRequest) -> Iterator[mx.array | None]:
        """Yield sixteen-codebook frames; None is a sampled utterance EOS."""
        model = self._model
        reference = (
            mx.array(reference_pcm(request.ref_audio, SAMPLE_RATE))
            if request.ref_audio is not None
            else None
        )
        conditional = model._prompt_embeddings(
            request.text,
            voice="S0",
            instruct=request.instruction,
            ref_audio=reference,
            ref_text=request.ref_text,
        )
        unconditional = model._prompt_embeddings(
            request.text,
            voice="S0",
            instruct=None,
            ref_audio=reference,
            ref_text=request.ref_text,
        )
        cond_cache = model.backbone_model.make_cache()
        uncond_cache = model.backbone_model.make_cache()
        cond_hidden = model.backbone_model(
            input_embeddings=conditional, cache=cond_cache
        )[:, -1, :]
        uncond_hidden = model.backbone_model(
            input_embeddings=unconditional, cache=uncond_cache
        )[:, -1, :]
        sampler = Sampler(request.seed)
        for _ in range(request.max_frames):
            scores = guidance(
                model.lm_head(cond_hidden),
                model.lm_head(uncond_hidden),
                request.cfg_scale,
            )
            first = sampler.sample(
                backbone_scores(
                    scores,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                )
            )
            if first.item() == EOS_TOKEN:
                yield None
                return
            depth = model.depth_decoder
            cond_depth = depth.model.make_cache()
            uncond_depth = depth.model.make_cache()
            depth.start_frame(cond_hidden, cond_depth)
            depth.start_frame(uncond_hidden, uncond_depth)
            token = first
            tokens = [first.reshape(1)]
            for head in range(CODEBOOKS - 1):
                cond_logits = depth.step_logits(
                    cond_depth, head_idx=head, token_id=token
                )
                uncond_logits = depth.step_logits(
                    uncond_depth, head_idx=head, token_id=token
                )
                token = sampler.sample(
                    depth_scores(
                        guidance(cond_logits, uncond_logits, request.cfg_scale),
                        temperature=request.temperature,
                        top_k=request.top_k,
                        top_p=request.top_p,
                    )
                )
                tokens.append(token.reshape(1))
            frame = mx.concatenate(tokens)
            mx.eval(frame)
            yield frame
            codes = frame[None, None, :]
            cond_hidden = model.backbone_model(input_ids=codes, cache=cond_cache)[
                :, -1, :
            ]
            uncond_hidden = model.backbone_model(input_ids=codes, cache=uncond_cache)[
                :, -1, :
            ]
