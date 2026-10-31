"""Reference Nemotron backend using the pinned MLX-VLM implementation."""

import platform
import wave
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from voicelab_models.nemotron.tools import ToolSettings

if TYPE_CHECKING:
    from voicelab_inference.nemotron.mlx.tool_result import ToolResult

from voicelab_core.contracts import GenerationSettings, SpeechOutput
from voicelab_inference.nemotron.mlx.streaming import NemotronStream
from voicelab_models.nemotron.model import OUTPUT_SAMPLE_RATE, Checkpoint


class NemotronMLX:
    def __init__(self, checkpoint: Checkpoint) -> None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("This backend requires an Apple Silicon Mac.")
        import mlx.core as mx
        from mlx_vlm import load
        from mlx_vlm.models.nemotron_voicechat import Model

        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal is unavailable.")
        model, processor = load(checkpoint.repository, revision=checkpoint.revision)
        if not isinstance(model, Model):
            raise TypeError("Checkpoint did not load a Nemotron VoiceChat model.")
        self._session = model.create_session(processor)
        self.synchronize()

    def start_stream(self, prompt: str, seed: int = 0) -> NemotronStream:
        return NemotronStream(self._session, prompt, seed)

    def synchronize(self) -> None:
        import mlx.core as mx

        mx.synchronize()

    def generate_with_tool(
        self,
        source: Path,
        destination: Path,
        settings: GenerationSettings,
        tools: ToolSettings,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ToolResult:
        from voicelab_inference.nemotron.mlx.tools import (
            generate_with_response,
            never_cancel,
        )

        return generate_with_response(
            self._session,
            source,
            destination,
            settings,
            tools,
            should_cancel or never_cancel,
        )

    def generate(
        self, source: Path, destination: Path, settings: GenerationSettings
    ) -> SpeechOutput:
        import mlx.core as mx
        import numpy as np

        if not settings.prompt.isascii():
            raise ValueError("Nemotron system prompts must be ASCII.")
        from voicelab_inference.nemotron.mlx.sampling import preserved

        with preserved():
            result = self._session.generate(
                str(source),
                system_prompt=settings.prompt,
                seed=settings.seed,
                extra_decoding_seconds=settings.trailing_silence_seconds,
            )
        mx.eval(result.audio, result.function_tokens)
        samples = np.asarray(result.audio, dtype=np.float32).reshape(-1)
        if samples.size == 0 or not np.isfinite(samples).all():
            raise RuntimeError("Backend produced empty or non-finite audio.")
        if result.sample_rate != OUTPUT_SAMPLE_RATE:
            raise RuntimeError("Unexpected Nemotron output sample rate.")
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
        with wave.open(str(destination), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(result.sample_rate)
            audio.writeframes(pcm.tobytes())
        tokens = tuple(
            int(token) for token in np.asarray(result.function_tokens).reshape(-1)
        )
        return SpeechOutput(
            destination, result.text, result.user_transcript, result.sample_rate, tokens
        )
