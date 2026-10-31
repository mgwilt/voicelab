"""Resident synthesis, bounded delivery, and cancellation for typed model requests."""

import queue
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from voicelab_core.synthesis import SynthesisMetrics
from voicelab_web.transport.pcm import Audio, pcm16


class Synthesizer[Request: BaseModel](Protocol):
    def synthesize(
        self, request: Request, emit: Callable[[Audio], None]
    ) -> SynthesisMetrics: ...


@runtime_checkable
class Closable(Protocol):
    def close(self) -> None: ...


@dataclass(frozen=True)
class Failure:
    message: str


type Event = bytes | SynthesisMetrics | Failure


class Delivery:
    def __init__(self) -> None:
        self.events: queue.Queue[Event] = queue.Queue(maxsize=8)
        self.cancelled = threading.Event()

    def publish(self, event: Event) -> None:
        while not self.cancelled.is_set():
            try:
                self.events.put(event, timeout=0.2)
                return
            except queue.Full:
                continue
        raise InterruptedError("Synthesis cancelled")

    def emit(self, audio: Audio) -> None:
        self.publish(pcm16(audio))

    def poll(self) -> Event | None:
        try:
            return self.events.get(timeout=0.2)
        except queue.Empty:
            return None


class SynthesisService[Request: BaseModel]:
    def __init__(
        self,
        request_type: type[Request],
        factory: Callable[[], Synthesizer[Request]],
        sample_rate: int = 24000,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("Invalid synthesis sample rate")
        self.request_type = request_type
        self.factory = factory
        self.sample_rate = sample_rate
        self.runtime: Synthesizer[Request] | None = None
        self.lock = threading.Lock()

    def unload(self) -> None:
        """Release residency only after the session worker has released its lock."""
        with self.lock:
            runtime = self.runtime
            self.runtime = None
            if isinstance(runtime, Closable):
                runtime.close()

    def generate(self, request: Request, delivery: Delivery) -> None:
        if not self.lock.acquire(blocking=False):
            with suppress(InterruptedError):
                delivery.publish(Failure("Another synthesis is in progress"))
            return
        try:
            if self.runtime is None:
                self.runtime = self.factory()
            if delivery.cancelled.is_set():
                return
            result = self.runtime.synthesize(request, delivery.emit)
            delivery.publish(result)
        except InterruptedError:
            pass
        except Exception as exc:
            with suppress(InterruptedError):
                delivery.publish(Failure(str(exc)))
        finally:
            self.lock.release()
