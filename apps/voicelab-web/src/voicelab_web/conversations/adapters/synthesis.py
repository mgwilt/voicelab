"""Reuse resident typed synthesis services without a second model registry."""

from pydantic import BaseModel

from voicelab.conversations.contracts import Audio, Cancelled, Emit
from voicelab_core.synthesis import SynthesisMetrics
from voicelab_web.models.synthesis.service import SynthesisService
from voicelab_web.transport.pcm import pcm16


class SynthesisVoice[Request: BaseModel]:
    def __init__(self, service: SynthesisService[Request], settings: BaseModel) -> None:
        self.service = service
        self.template = service.request_type.model_validate(settings.model_dump())

    @property
    def sample_rate(self) -> int:
        return self.service.sample_rate

    def speak(self, text: str, emit: Emit, cancelled: Cancelled) -> SynthesisMetrics:
        from voicelab_web.transport.pcm import Audio as Samples

        request = self.service.request_type.model_validate(
            {**self.template.model_dump(), "text": text}
        )

        def deliver(samples: Samples) -> None:
            if cancelled():
                raise InterruptedError("Speech interrupted")
            emit(Audio(pcm16(samples), self.service.sample_rate))

        with self.service.lock:
            if self.service.runtime is None:
                self.service.runtime = self.service.factory()
            if cancelled():
                raise InterruptedError("Speech interrupted")
            return self.service.runtime.synthesize(request, deliver)

    def close(self) -> None:
        self.service.unload()
