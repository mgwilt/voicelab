"""Own the model and its sessions on one dedicated inference thread."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from voicelab_core.contracts import GenerationSettings
from voicelab_models.nemotron.tools import ToolSettings
from voicelab_web.transport.worker import InferenceWorker

if TYPE_CHECKING:
    from voicelab_inference.nemotron.mlx.tool_result import ToolResult

from voicelab_inference.nemotron.mlx.runtime import NemotronMLX
from voicelab_models.nemotron.model import MLX_8BIT
from voicelab_models.nemotron.streaming import DuplexSession, StreamEvent


class Engine:
    def __init__(self) -> None:
        self._worker = InferenceWorker("nemotron")
        self._model: NemotronMLX | None = None
        self._session: DuplexSession | None = None
        self.busy = False

    async def call[T](self, operation: Callable[[], T]) -> T:
        return await self._worker.call(operation)

    def start(self, prompt: str, seed: int) -> float:
        start = time.perf_counter()
        if self._model is None:
            self._model = NemotronMLX(MLX_8BIT)
        self._session = self._model.start_stream(prompt, seed)
        self._model.synchronize()
        return time.perf_counter() - start

    def run_tool(
        self,
        source: Path,
        destination: Path,
        settings: GenerationSettings,
        tools: ToolSettings,
        should_cancel: Callable[[], bool],
    ) -> ToolResult:
        if self._model is None:
            self._model = NemotronMLX(MLX_8BIT)
        if should_cancel():
            raise InterruptedError("Nemotron tool session cancelled.")
        return self._model.generate_with_tool(
            source, destination, settings, tools, should_cancel
        )

    def push(self, pcm: bytes) -> tuple[tuple[StreamEvent, ...], float]:
        if self._session is None:
            raise RuntimeError("No active session.")
        start = time.perf_counter()
        events = self._session.push(pcm)
        return events, time.perf_counter() - start

    def close_session(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def shutdown(self) -> None:
        self._worker.shutdown()

    def unload(self) -> None:
        self.close_session()
        if self._model is not None:
            import mlx.core as mx

            self._model.synchronize()
            self._model = None
            mx.clear_cache()
