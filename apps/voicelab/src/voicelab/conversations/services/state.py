"""Per-call models, bounded context, and serialized accelerator ownership."""

import asyncio
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol

from voicelab.conversations.contracts import Notice, Notify, Voice
from voicelab.conversations.history import History
from voicelab.runs.worker import InferenceWorker
from voicelab_inference.qwen_text.mlx.runtime import Message


class Transcriber(Protocol):
    def load(self) -> None: ...
    def transcribe(self, pcm: bytes) -> str: ...
    def close(self) -> None: ...


class Generator(Protocol):
    def load(self) -> None: ...
    def generate(
        self, messages: tuple[Message, ...], cancelled: Callable[[], bool]
    ) -> Iterator[str]: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class Turn:
    number: int
    cancelled: threading.Event


class CascadeState:
    def __init__(
        self,
        stt: Transcriber,
        llm: Generator,
        voice: Voice,
        system_prompt: str,
        notify: Notify,
    ) -> None:
        self.stt, self.llm, self.voice = stt, llm, voice
        self.system = Message("system", system_prompt)
        self.notify = notify
        self.worker = InferenceWorker("cascade")
        self.history = History[Message](limit=12)
        self.turn = Turn(0, threading.Event())
        self.stopping = False
        self.user: Message | None = None
        self.answer = ""
        self.generating = False
        self.synthesizing = False
        self.output_pending = False
        self.response_started = 0.0
        self.close_task: asyncio.Task[None] | None = None

    def interrupt(self) -> None:
        self.turn.cancelled.set()
        self.turn = Turn(self.turn.number + 1, threading.Event())
        self.history.interrupt()
        self.user = None
        self.answer = ""
        self.output_pending = False
        self.generating = self.synthesizing = False

    def current(self, turn: Turn) -> bool:
        return not self.stopping and not turn.cancelled.is_set() and turn is self.turn

    async def notice(self, notice: Notice, turn: Turn | None = None) -> None:
        if turn is None or self.current(turn):
            await self.notify(
                notice.model_copy(update={"turn": (turn or self.turn).number})
            )

    async def stream[T](
        self,
        operation: Callable[[Callable[[T], None]], None],
        turn: Turn,
    ) -> AsyncIterator[T]:
        """Bridge bounded model callbacks without blocking the event loop."""
        if not self.current(turn):
            return
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[T] = asyncio.Queue(maxsize=8)

        def emit(value: T) -> None:
            if not self.current(turn):
                raise InterruptedError("Conversation interrupted")
            pending = asyncio.run_coroutine_threadsafe(queue.put(value), loop)
            try:
                while self.current(turn):
                    try:
                        pending.result(timeout=0.1)
                        return
                    except FutureTimeout:
                        continue
                raise InterruptedError("Conversation interrupted")
            finally:
                if not pending.done():
                    pending.cancel()

        job = asyncio.create_task(self.worker.call(lambda: operation(emit)))
        try:
            while self.current(turn) and (not job.done() or not queue.empty()):
                item = asyncio.create_task(queue.get())
                try:
                    await asyncio.wait((item, job), return_when=asyncio.FIRST_COMPLETED)
                    if item.done():
                        yield item.result()
                    elif job.done():
                        if queue.empty():
                            break
                        yield await item
                finally:
                    if not item.done():
                        item.cancel()
                    with suppress(asyncio.CancelledError):
                        await item
            await job
        finally:
            if not job.done():
                turn.cancelled.set()
                job.cancel()
            with suppress(asyncio.CancelledError, InterruptedError):
                await job

    async def close(self) -> None:
        if self.close_task is None:
            self.stopping = True
            self.turn.cancelled.set()
            self.close_task = asyncio.create_task(self._close())
        await asyncio.shield(self.close_task)

    async def _close(self) -> None:
        def release() -> None:
            try:
                self.voice.close()
            finally:
                try:
                    self.llm.close()
                finally:
                    self.stt.close()

        try:
            await self.worker.call(release)
        finally:
            self.worker.shutdown()
