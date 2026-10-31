"""Serialized resident Breeze runtime with explicit lazy loading."""

import platform
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from voicelab_core.synthesis import SynthesisMetrics
from voicelab_inference.breeze.mlx.audio import Audio
from voicelab_models.breeze.synthesis import SynthesisRequest

if TYPE_CHECKING:
    from voicelab_inference.breeze.mlx.vendor import BreezeModel


class BreezeMLX:
    implementation = "breeze-mlx/native-streaming"
    precision = "bfloat16 language model; float32 codec weights"
    device = "mlx"

    def __init__(self) -> None:
        self._model: BreezeModel | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            self._load()

    def close(self) -> None:
        with self._lock:
            if self._model is not None:
                import mlx.core as mx

                mx.synchronize()
                self._model = None
                mx.clear_cache()

    def _load(self) -> None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Breeze MLX requires Apple Silicon macOS")
        if self._model is None:
            from voicelab_inference.breeze.mlx.vendor import BreezeModel

            self._model = BreezeModel()

    def synthesize(
        self, request: SynthesisRequest, emit: Callable[[Audio], None]
    ) -> SynthesisMetrics:
        from voicelab_inference.breeze.mlx.stream import synthesize

        with self._lock:
            self._load()
            if self._model is None:
                raise RuntimeError("Breeze did not load")
            return synthesize(self._model, request, emit)
