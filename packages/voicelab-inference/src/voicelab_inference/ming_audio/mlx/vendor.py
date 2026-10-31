"""Pinned native MLX thinker and talker, with separate completion signals."""

from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_audio.sts.models.ming_v3_audio.ming_v3_audio import Model
from mlx_audio.sts.models.ming_v3_audio.talker import Talker
from mlx_audio.sts.utils import load_model
from transformers import PreTrainedTokenizerBase

from voicelab_inference.ming_audio.mlx.answer import parse_answer
from voicelab_inference.ming_audio.mlx.contracts import Output
from voicelab_inference.ming_audio.mlx.decoding import decode
from voicelab_inference.ming_audio.mlx.messages import messages
from voicelab_inference.ming_audio.mlx.sampling import seeded
from voicelab_inference.ming_audio.mlx.speech import synthesize
from voicelab_models.ming_audio.model import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    Request,
)


def check_cancel(should_cancel: Callable[[], bool]) -> None:
    if should_cancel():
        raise InterruptedError("Audio conversation cancelled")


class Vendor:
    def __init__(self) -> None:
        with seeded(0):
            self._initialize()

    def _initialize(self) -> None:
        path = Path(
            snapshot_download(
                CHECKPOINT_ID,
                revision=CHECKPOINT_REVISION,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.jinja",
                    "*.txt",
                    "*.model",
                ],
            )
        )
        model = load_model(path, strict=True)
        if not isinstance(model, Model):
            raise TypeError("Checkpoint did not load a Ming Audio model")
        talker = model._get_talker()
        if not isinstance(talker, Talker) or talker.sample_rate != OUTPUT_SAMPLE_RATE:
            raise ValueError("Checkpoint must contain a 44.1 kHz Ming talker")
        tokenizer = model.tokenizer
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError("Ming tokenizer did not load")
        self.tokenizer = tokenizer
        self.model = model
        self.talker = talker

    def respond(self, request: Request, should_cancel: Callable[[], bool]) -> Output:
        with seeded(request.seed):
            return self._respond(request, should_cancel)

    def _respond(self, request: Request, should_cancel: Callable[[], bool]) -> Output:
        check_cancel(should_cancel)
        try:
            reasoning: str | None = None
            language: str | None = None
            text: str
            if request.task == "tts":
                text, thinker_eos = request.prompt, True
            else:
                _, embeddings = self.model.prepare_messages(
                    messages(request),
                    thinking=request.thinking,
                    sample_rate=INPUT_SAMPLE_RATE,
                )
                tokens, thinker_eos = decode(
                    self.model, embeddings, request.max_tokens, should_cancel
                )
                parsed = parse_answer(
                    self.tokenizer.decode(tokens, skip_special_tokens=False),
                    thinking=request.thinking,
                    transcription=request.task == "asr",
                )
                text, reasoning, language = (
                    parsed.text,
                    parsed.reasoning,
                    parsed.language,
                )
                thinker_eos = thinker_eos and parsed.reasoning_complete
            if not text and thinker_eos:
                raise ValueError("Ming Thinker returned no answer")
            check_cancel(should_cancel)
            audio = None
            completed = thinker_eos
            if request.task in ("spoken_dialogue", "tts") and text:
                audio, talker_eos = synthesize(
                    self.talker, text, request, should_cancel
                )
                completed = completed and talker_eos
            return Output(
                text=text,
                audio=audio,
                completed=completed,
                reasoning=reasoning,
                language=language,
            )
        finally:
            mx.clear_cache()

    def close(self) -> None:
        mx.clear_cache()
