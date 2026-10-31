"""Qwen's native audio turns and bounded audio/text history."""

from voicelab.conversations.audio import reference
from voicelab.conversations.contracts import Audio, Cancelled, Emit, Notice
from voicelab.conversations.history import History
from voicelab_inference.qwen_omni.mlx.runtime import QwenOmniMLX
from voicelab_models.qwen_omni.model import HistoryTurn, Request


class QwenTurns:
    input_rate = 16000
    frame_bytes = 0
    duplex = False

    def __init__(self, settings: Request) -> None:
        self.settings = settings
        self.runtime = QwenOmniMLX()
        self.history = History(settings.history)

    def load(self, emit: Emit, cancelled: Cancelled) -> None:
        self.runtime.load()

    def process(self, pcm: bytes, emit: Emit, cancelled: Cancelled) -> None:
        audio = reference(pcm)
        request = Request.model_validate(
            {
                **self.settings.model_dump(),
                "audio": audio,
                "history": self.history.snapshot(),
                "return_audio": True,
            }
        )
        emit(Notice(type="text", speaker="user", text="Audio message", final=True))
        response = self.runtime.respond(
            request,
            lambda text: emit(Notice(type="text", text=text)),
            lambda chunk: emit(Audio(chunk, 24000)),
            cancelled,
        )
        emit(Notice(type="text", text=response.text, final=True))
        emit(
            Notice(
                type="metrics",
                stage="response",
                text="complete" if response.metrics.completed else "incomplete",
                seconds=response.metrics.generation_seconds,
                output_rtf=response.metrics.rtf,
            )
        )
        if response.metrics.completed and not cancelled():
            self.history.stage(
                HistoryTurn(role="user", content=request.prompt, audio=audio),
                HistoryTurn(role="assistant", content=response.text),
            )

    def played(self) -> None:
        self.history.played()

    def interrupt(self) -> None:
        self.history.interrupt()

    def close(self) -> None:
        self.runtime.close()
