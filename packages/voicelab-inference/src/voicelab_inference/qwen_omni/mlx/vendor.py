"""Pinned local Thinker, Talker, and Code2Wav composition."""

from collections.abc import Callable

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)
from mlx_vlm.models.qwen3_omni_moe.qwen3_omni_moe import Model
from numpy.typing import NDArray
from transformers import PreTrainedTokenizerBase

from voicelab_inference.qwen_omni.mlx import conditioning, inputs, speech, thinker
from voicelab_inference.qwen_omni.mlx.contracts import Completion
from voicelab_inference.qwen_omni.mlx.sampling import check_cancel, seeded
from voicelab_models.qwen_omni.model import CHECKPOINT_ID, CHECKPOINT_REVISION, Request


class Vendor:
    def __init__(self) -> None:
        with seeded(0):
            model, processor = load(
                CHECKPOINT_ID, revision=CHECKPOINT_REVISION, strict=True
            )
            if (
                not isinstance(model, Model)
                or model.talker is None
                or model.code2wav is None
            ):
                raise TypeError(
                    "Qwen checkpoint must include Thinker, Talker, and Code2Wav"
                )
            if not isinstance(processor, Qwen3OmniMoeProcessor):
                raise TypeError(
                    "Checkpoint did not load the native Qwen Omni processor"
                )
            tokenizer = processor.tokenizer
            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise TypeError("Checkpoint tokenizer is invalid")
            self.model = model
            self.processor = processor
            self.tokenizer = tokenizer

    def generate(
        self,
        request: Request,
        emit_text: Callable[[str], None],
        emit_audio: Callable[[NDArray[np.float32]], None],
        should_cancel: Callable[[], bool],
    ) -> Completion:
        with seeded(request.seed):
            try:
                check_cancel(should_cancel)
                prepared = inputs.prepare(self.processor, request)
                thought = thinker.decode(
                    self.model,
                    self.tokenizer,
                    prepared,
                    request,
                    emit_text,
                    should_cancel,
                )
                eos: bool | None = None
                frames = 0
                if request.return_audio and thought.text.strip():
                    context = conditioning.prepare(
                        self.model, prepared.ids, thought, request.speaker
                    )
                    eos, frames = speech.generate(
                        self.model, context, request, emit_audio, should_cancel
                    )
                elif request.return_audio:
                    eos = False
                return Completion(
                    thought.text, thought.eos, eos, thought.tokens, frames
                )
            finally:
                mx.clear_cache()

    def close(self) -> None:
        mx.synchronize()
        mx.clear_cache()
