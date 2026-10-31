"""Own the model and its sessions on one dedicated inference thread."""

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from voicelab_inference.nemotron.mlx.runtime import NemotronMLX
from voicelab_models.nemotron.model import MLX_8BIT
from voicelab_models.nemotron.streaming import DuplexSession, StreamEvent


class Engine:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="inference"
        )
        self._model: NemotronMLX | None = None
        self._session: DuplexSession | None = None
        self.busy = False

    async def call[T](self, operation: Callable[[], T]) -> T:
        future = asyncio.get_running_loop().run_in_executor(self._executor, operation)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # Cancellation cannot stop a GPU kernel: wait before releasing the lease.
            await future
            raise

    def start(self, prompt: str, seed: int) -> float:
        start = time.perf_counter()
        if self._model is None:
            self._model = NemotronMLX(MLX_8BIT)
        self._session = self._model.start_stream(prompt, seed)
        self._model.synchronize()
        return time.perf_counter() - start

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
        self._executor.shutdown(wait=True)
