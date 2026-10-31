"""Ming's buffered native speech responses and session history."""

from voicelab.conversations.audio import reference
from voicelab.conversations.contracts import Audio, Cancelled, Emit, Notice
from voicelab.conversations.history import History
from voicelab_inference.ming_audio.mlx.runtime import MingAudioMLX
from voicelab_models.ming_audio.model import HistoryTurn, Request


class MingTurns:
    input_rate = 16000
    frame_bytes = 0
    duplex = False

    def __init__(self, settings: Request) -> None:
        self.settings = settings
        self.runtime = MingAudioMLX()
        self.history = History(settings.history)

    def load(self, emit: Emit, cancelled: Cancelled) -> None:
        self.runtime.load()

    def process(self, pcm: bytes, emit: Emit, cancelled: Cancelled) -> None:
        audio = reference(pcm)
        request = Request.model_validate(
            {
                **self.settings.model_dump(),
                "task": "spoken_dialogue",
                "audio_wav_base64": audio.wav_base64,
                "history": self.history.snapshot(),
            }
        )
        emit(Notice(type="text", speaker="user", text="Audio message", final=True))
        result = self.runtime.respond(request, cancelled)
        emit(Notice(type="text", text=result.text, final=True))
        if result.pcm:
            emit(Audio(result.pcm, 44100))
        emit(
            Notice(
                type="metrics",
                stage="response",
                text="complete" if result.metrics.completed else "incomplete",
                seconds=result.metrics.generation_seconds,
                output_rtf=result.metrics.rtf,
            )
        )
        if result.metrics.completed and not cancelled():
            self.history.stage(
                HistoryTurn(role="user", content=request.prompt, audio=audio),
                HistoryTurn(role="assistant", content=result.text),
            )

    def played(self) -> None:
        self.history.played()

    def interrupt(self) -> None:
        self.history.interrupt()

    def close(self) -> None:
        self.runtime.close()
