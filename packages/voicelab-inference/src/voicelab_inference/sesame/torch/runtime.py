"""One resident CSM model; generation owns the codec and random state exclusively."""

import gc
import threading
import types
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import CsmForConditionalGeneration, CsmProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import BatchEncoding

from voicelab_core.synthesis import SynthesisMetrics
from voicelab_inference.audio import reference_pcm
from voicelab_inference.sesame.torch.audio import Audio, validate_audio
from voicelab_inference.sesame.torch.decoding.depth import direct_depth_generate
from voicelab_inference.sesame.torch.decoding.mimi import StatefulMimi
from voicelab_inference.sesame.torch.decoding.stream import AudioStreamer
from voicelab_inference.sesame.torch.random import random_state
from voicelab_models.sesame.model import (
    CONTEXT_LENGTH,
    MODEL_ID,
    REVISION,
    Device,
    Engine,
    SynthesisRequest,
)


@dataclass(frozen=True)
class ReferenceSpeech:
    samples: Audio
    transcript: str

    def __post_init__(self) -> None:
        validate_audio(self.samples)
        if not self.transcript.strip():
            raise ValueError("Reference transcript must not be empty")
        if self.samples.size > 24_000 * 30:
            raise ValueError("Reference audio exceeds 30 seconds")


def synchronize(device: Device) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


class ResidentCSM:
    def __init__(self, device: Device, engine: Engine = "direct-depth") -> None:
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        self.device = device
        self.engine = engine
        self.precision = "float32" if device == "cpu" else "bfloat16"
        self._lock = threading.Lock()
        self.processor = CsmProcessor.from_pretrained(MODEL_ID, revision=REVISION)
        # CPU materialization avoids direct sharded MPS loading failures.
        model = CsmForConditionalGeneration.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            device_map="cpu",
            attn_implementation="sdpa",
        )
        if not isinstance(model, CsmForConditionalGeneration):
            raise TypeError("Unexpected checkpoint architecture")
        self._model: CsmForConditionalGeneration | None = model.eval().to(device)
        self.model.codec_model.to(dtype=torch.float32)
        if engine == "direct-depth":
            self.model.depth_decoder.generate = types.MethodType(
                direct_depth_generate, self.model.depth_decoder
            )
        synchronize(device)

    @property
    def model(self) -> CsmForConditionalGeneration:
        if self._model is None:
            raise RuntimeError("CSM has been closed")
        return self._model

    def close(self) -> None:
        with self._lock:
            if self._model is None:
                return
            synchronize(self.device)
            self._model = None
            # Bound decoder methods can retain cycles until collection.
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()

    def _inputs(
        self, request: SynthesisRequest, reference: ReferenceSpeech | None
    ) -> BatchFeature | BatchEncoding:
        if reference is None and not request.context:
            inputs = self.processor(
                f"[{request.speaker}]{request.text}",
                add_special_tokens=True,
                return_tensors="pt",
            )
        else:
            # Vendor serialization boundary; domain data stays typed above it.
            history = [
                (segment.speaker, segment.text, reference_pcm(segment.audio, 24_000))
                for segment in request.context
            ]
            if reference is not None:
                if history:
                    raise ValueError("Use reference speech or conversation context")
                history.append(
                    (request.speaker, reference.transcript, reference.samples)
                )
            conversation = [
                {
                    "role": str(speaker),
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "audio", "path": samples},
                    ],
                }
                for speaker, text, samples in history
            ] + [
                {
                    "role": str(request.speaker),
                    "content": [
                        {"type": "text", "text": request.text},
                    ],
                },
            ]
            inputs = self.processor.apply_chat_template(
                # CSM documents audio content; vendor annotation is text-only.
                conversation,  # ty: ignore[invalid-argument-type]
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        if not isinstance(inputs, (BatchFeature, BatchEncoding)):
            raise TypeError("Unexpected processor output")
        ids = inputs["input_ids"]
        if not isinstance(ids, Tensor) or ids.ndim != 2:
            raise ValueError("Processor returned invalid token IDs")
        if ids.shape[1] + request.max_frames > CONTEXT_LENGTH:
            raise ValueError(
                "Text, reference, and audio budget exceed context capacity"
            )
        return inputs.to(self.device)

    @torch.inference_mode()
    def synthesize(
        self,
        request: SynthesisRequest,
        emit: Callable[[Audio], None],
        reference: ReferenceSpeech | None = None,
    ) -> SynthesisMetrics:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("CSM is already generating")
        try:
            with random_state(self.device, request.seed):
                return self._synthesize(request, emit, reference)
        finally:
            self._lock.release()

    def _synthesize(
        self,
        request: SynthesisRequest,
        emit: Callable[[Audio], None],
        reference: ReferenceSpeech | None,
    ) -> SynthesisMetrics:
        stream: AudioStreamer | None = None
        try:
            if request.ref_audio is not None and request.ref_text is not None:
                if reference is not None:
                    raise ValueError("Provide one reference source, not both")
                reference = ReferenceSpeech(
                    reference_pcm(request.ref_audio, 24_000), request.ref_text
                )
            inputs = self._inputs(request, reference)
            synchronize(self.device)
            codebooks = self.model.config.num_codebooks
            eos_token = self.model.config.codebook_eos_token_id
            if not isinstance(codebooks, int) or not isinstance(eos_token, int):
                raise ValueError("CSM requires integer codebook count and EOS token")
            stream = AudioStreamer(
                StatefulMimi(self.model.codec_model),
                emit,
                device=self.device,
                codebooks=codebooks,
                eos_token=eos_token,
                max_frames=request.max_frames,
            )
            self.model.generate(
                **inputs,
                output_audio=False,
                max_new_tokens=request.max_frames,
                do_sample=True,
                temperature=request.temperature,
                top_k=request.top_k,
                streamer=stream,
            )
            synchronize(self.device)
            stream.end()
            return stream.metrics()
        finally:
            if stream is not None:
                stream.close()
