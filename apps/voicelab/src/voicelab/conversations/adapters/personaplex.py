"""Preserve PersonaPlex's native 24 kHz duplex clock."""

from voicelab.conversations.contracts import Audio, Cancelled, Emit, Notice
from voicelab_inference.personaplex.mlx.runtime import PersonaPlexMLX
from voicelab_models.personaplex.model import Request


class PersonaPlexDuplex:
    input_rate = 24000
    frame_bytes = 3840
    duplex = True

    def __init__(self, settings: Request) -> None:
        self.settings = settings
        self.runtime = PersonaPlexMLX()

    def load(self, emit: Emit, cancelled: Cancelled) -> None:
        self.runtime.start(self.settings)

    def process(self, pcm: bytes, emit: Emit, cancelled: Cancelled) -> None:
        for event in self.runtime.push(pcm):
            if event.kind == "audio":
                emit(Audio(event.pcm, 24000))
            else:
                emit(Notice(type="text", text=event.text))
        emit(
            Notice(type="metrics", stage="duplex", input_rtf=self.runtime.metrics().rtf)
        )

    def played(self) -> None:
        # The model owns its recurrent conversation state.
        pass

    def interrupt(self) -> None:
        # Native duplex handles simultaneous listening and speaking.
        pass

    def close(self) -> None:
        self.runtime.close()
