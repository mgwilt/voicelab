"""Bounded speech delivery from one inference worker to its socket."""

import queue
import threading
from dataclasses import dataclass

from voicelab_inference.qwen_omni.mlx.contracts import Metrics


@dataclass(frozen=True)
class Text:
    value: str


@dataclass(frozen=True)
class Ready:
    setup_seconds: float


type Event = Text | Ready | bytes | Metrics


class Delivery:
    def __init__(self) -> None:
        self.events: queue.Queue[Event] = queue.Queue(maxsize=8)
        self.cancelled = threading.Event()

    def put(self, event: Event) -> None:
        while not self.cancelled.is_set():
            try:
                self.events.put(event, timeout=0.1)
                return
            except queue.Full:
                continue
        raise InterruptedError("Qwen conversation cancelled")

    def text(self, value: str) -> None:
        if value:
            self.put(Text(value))

    def audio(self, pcm: bytes) -> None:
        for index in range(0, len(pcm), 32768):
            self.put(pcm[index : index + 32768])

    def poll(self) -> Event | None:
        try:
            return self.events.get(timeout=0.1)
        except queue.Empty:
            return None
