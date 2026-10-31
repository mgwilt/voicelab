"""Managed local transcription, text generation, and selected voice synthesis."""

from time import perf_counter

from voicelab_inference.qwen_text.mlx.runtime import Message, QwenTextMLX
from voicelab_inference.whisper.mlx.runtime import WhisperMLX
from voicelab_web.conversations.adapters.synthesis import Voice
from voicelab_web.conversations.contracts import Cancelled, Emit, Notice
from voicelab_web.conversations.history import History


class Cascade:
    input_rate = 16000
    frame_bytes = 0
    duplex = False

    def __init__(self, voice: Voice, system_prompt: str) -> None:
        self.voice = voice
        self.stt = WhisperMLX()
        self.llm = QwenTextMLX()
        self.system = Message("system", system_prompt)
        self.history: History[Message] = History(limit=12)

    def load(self, emit: Emit, cancelled: Cancelled) -> None:
        emit(Notice(type="state", text="Loading local Whisper"))
        self.stt.load()
        if cancelled():
            raise InterruptedError("Loading cancelled")
        emit(Notice(type="state", text="Loading local Qwen"))
        self.llm.load()

    def process(self, pcm: bytes, emit: Emit, cancelled: Cancelled) -> None:
        emit(Notice(type="state", text="Transcribing"))
        start = perf_counter()
        text = self.stt.transcribe(pcm)
        emit(Notice(type="metrics", stage="stt", seconds=perf_counter() - start))
        if cancelled() or not text:
            return
        emit(Notice(type="text", speaker="user", text=text, final=True))
        emit(Notice(type="state", text="Thinking"))
        user = Message("user", text)
        start = perf_counter()
        answer = ""
        first = True
        for delta in self.llm.generate(
            (self.system, *self.history.snapshot(), user), cancelled
        ):
            if first:
                emit(
                    Notice(
                        type="metrics",
                        stage="llm_first_token",
                        seconds=perf_counter() - start,
                    )
                )
                first = False
            answer += delta
            emit(Notice(type="text", text=delta))
        emit(Notice(type="metrics", stage="llm", seconds=perf_counter() - start))
        if cancelled() or not answer.strip():
            return
        emit(Notice(type="text", text=answer, final=True))
        emit(Notice(type="state", text="Speaking"))
        # Preserve each model's synthesis semantics; no arbitrary sentence slicing.
        result = self.voice.speak(answer, emit, cancelled)
        emit(
            Notice(
                type="metrics",
                stage="tts",
                text=result.completion_state,
                seconds=result.generation_seconds,
                output_rtf=result.output_rtf,
            )
        )
        if not cancelled() and result.completion_state != "incomplete":
            self.history.stage(user, Message("assistant", answer))

    def played(self) -> None:
        self.history.played()

    def interrupt(self) -> None:
        self.history.interrupt()

    def close(self) -> None:
        try:
            self.voice.close()
        finally:
            try:
                self.llm.close()
            finally:
                self.stt.close()
