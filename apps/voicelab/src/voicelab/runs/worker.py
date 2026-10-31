"""One local inference thread; cancellation waits for submitted work to finish."""

import asyncio
import contextlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor


class InferenceWorker:
    def __init__(self, name: str) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)

    async def call[T](self, operation: Callable[[], T]) -> T:
        future = asyncio.get_running_loop().run_in_executor(self._executor, operation)
        cancelled = False
        while not future.done():
            try:
                # wait() never cancels the device future when its caller is cancelled.
                await asyncio.wait((future,))
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            # Consume the device error, but preserve the caller's cancellation.
            with contextlib.suppress(Exception):
                future.result()
            raise asyncio.CancelledError
        return future.result()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
