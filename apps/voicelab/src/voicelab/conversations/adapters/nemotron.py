"""Preserve Nemotron's native 80 ms recurrent audio timeline."""

from time import perf_counter

from voicelab.conversations.contracts import Audio, Cancelled, Emit, Notice
from voicelab_inference.nemotron.mlx.runtime import NemotronMLX
from voicelab_models.nemotron.model import MLX_8BIT
from voicelab_models.nemotron.streaming import DuplexSession


class NemotronDuplex:
    input_rate = 16000
    frame_bytes = 2560
    duplex = True

    def __init__(self, prompt: str, seed: int) -> None:
        self.prompt = prompt
        self.seed = seed
        self.runtime: NemotronMLX | None = None
        self.session: DuplexSession | None = None
        self.elapsed = 0.0
        self.input_seconds = 0.0

    def load(self, emit: Emit, cancelled: Cancelled) -> None:
        self.runtime = NemotronMLX(MLX_8BIT)
        self.session = self.runtime.start_stream(self.prompt, self.seed)
        self.runtime.synchronize()

    def process(self, pcm: bytes, emit: Emit, cancelled: Cancelled) -> None:
        if self.session is None:
            raise RuntimeError("Nemotron session is not loaded")
        start = perf_counter()
        events = self.session.push(pcm)
        self.elapsed += perf_counter() - start
        self.input_seconds += len(pcm) / (self.input_rate * 2)
        for event in events:
            if event.kind == "audio":
                emit(Audio(event.pcm, 22050))
            elif event.kind in ("assistant_text_delta", "user_transcript_delta"):
                emit(
                    Notice(
                        type="text",
                        text=event.text,
                        text_mode="snapshot",
                        speaker="user"
                        if event.kind == "user_transcript_delta"
                        else "assistant",
                    )
                )
        emit(
            Notice(
                type="metrics",
                stage="duplex",
                input_rtf=self.elapsed / self.input_seconds,
            )
        )

    def played(self) -> None:
        # The model owns its recurrent conversation state.
        pass

    def interrupt(self) -> None:
        # Listening/speaking decisions belong to the native recurrent model.
        pass

    def close(self) -> None:
        if self.session:
            self.session.close()
            self.session = None
        if self.runtime:
            import mlx.core as mx

            self.runtime.synchronize()
            self.runtime = None
            mx.clear_cache()
