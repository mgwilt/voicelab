"""MiniCPM native duplex or audio turns with transcribed text history."""

import threading

from voicelab_inference.minicpm.native.protocol import Event
from voicelab_inference.minicpm.native.runtime import MiniCPMNative
from voicelab_inference.whisper.mlx.runtime import WhisperMLX
from voicelab_models.minicpm.model import HistoryTurn, Request
from voicelab_web.conversations.audio import reference
from voicelab_web.conversations.contracts import Audio, Cancelled, Emit, Notice
from voicelab_web.conversations.history import History


class MiniCPMConversation:
    input_rate = 16000

    def __init__(self, settings: Request, *, duplex: bool) -> None:
        self.duplex = duplex
        self.frame_bytes = 32000 if duplex else 0
        self.settings = Request.model_validate(
            {
                **settings.model_dump(),
                "mode": "full_duplex" if duplex else "turn_based",
                "text": "" if duplex else "Listen to the audio.",
                "audio": None,
                "history": () if duplex else settings.history,
            }
        )
        self.runtime = MiniCPMNative()
        self.stt = None if duplex else WhisperMLX()
        self.history = History(self.settings.history)
        self.restart = threading.Event()

    def load(self, emit: Emit, cancelled: Cancelled) -> None:
        if self.stt:
            emit(Notice(type="state", text="Loading Whisper for MiniCPM text history"))
            self.stt.load()
        emit(Notice(type="state", text="Loading MiniCPM"))
        self.runtime.start(self.settings, cancelled)

    def process(self, pcm: bytes, emit: Emit, cancelled: Cancelled) -> None:
        answer = ""

        def deliver(event: Event) -> None:
            nonlocal answer
            if event.kind == "audio":
                emit(Audio(event.pcm, 24000))
            elif event.kind == "text":
                answer += event.text
                emit(Notice(type="text", text=event.text))

        if self.duplex:
            self.runtime.push(pcm, deliver, cancelled)
        else:
            if self.restart.is_set():
                self.runtime.close()
                self.runtime.start(self.settings, cancelled)
                self.restart.clear()
            if self.stt is None:
                raise RuntimeError("MiniCPM history transcriber is not loaded")
            text = self.stt.transcribe(pcm)
            emit(
                Notice(
                    type="text",
                    speaker="user",
                    text=text or "Audio message",
                    final=True,
                )
            )
            request = Request.model_validate(
                {
                    **self.settings.model_dump(),
                    "text": "",
                    "audio": reference(pcm),
                    "history": self.history.snapshot(),
                }
            )
            self.runtime.turn(deliver, cancelled, request=request)
            if (
                not cancelled()
                and text
                and answer
                and self.runtime.metrics().completed is True
            ):
                self.history.stage(
                    HistoryTurn(role="user", text=text[:1200]),
                    HistoryTurn(role="assistant", text=answer[:1200]),
                )
            emit(Notice(type="text", text=answer, final=True))
        metrics = self.runtime.metrics()
        emit(
            Notice(
                type="metrics",
                stage="duplex" if self.duplex else "response",
                input_rtf=metrics.rtf if self.duplex else None,
                output_rtf=None if self.duplex else metrics.rtf,
            )
        )

    def played(self) -> None:
        self.history.played()

    def interrupt(self) -> None:
        if not self.duplex:
            self.history.interrupt()
            self.restart.set()

    def close(self) -> None:
        try:
            self.runtime.close()
        finally:
            if self.stt:
                self.stt.close()
