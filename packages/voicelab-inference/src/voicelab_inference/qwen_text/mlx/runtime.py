"""Managed Qwen text generation; called on the session's accelerator worker."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

from voicelab_models.qwen_text.model import CHECKPOINT_ID, REVISION


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


class QwenTextMLX:
    def __init__(self) -> None:
        from mlx.nn import Module
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        self.model: Module | None = None
        self.tokenizer: TokenizerWrapper | None = None

    def load(self) -> None:
        from mlx_lm import load

        loaded = load(CHECKPOINT_ID, revision=REVISION)
        self.model, self.tokenizer = loaded[0], loaded[1]

    def generate(
        self, messages: tuple[Message, ...], cancelled: Callable[[], bool]
    ) -> Iterator[str]:
        from mlx_lm import stream_generate

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Qwen text is not loaded")
        prompt: object = self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(prompt, str):
            raise TypeError("Qwen tokenizer did not produce a text prompt")
        for response in stream_generate(
            self.model, self.tokenizer, prompt=prompt, max_tokens=256
        ):
            if cancelled():
                raise InterruptedError("Text generation interrupted")
            yield response.text

    def close(self) -> None:
        import mlx.core as mx

        self.model = self.tokenizer = None
        mx.clear_cache()
