"""Validated public model identities and transport capabilities."""

import platform
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

ModelId = Literal[
    "nemotron",
    "sesame",
    "breeze",
    "phonon",
    "kokoro",
    "audio8",
    "ming_tts",
    "drama",
    "maya",
    "omnivoice",
    "kitten",
    "personaplex",
    "ming_audio",
    "qwen_omni",
    "minicpm",
]


class ModelInfo(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    label: str
    description: str
    backend: str
    available: bool
    reason: str | None = None

    @computed_field
    @property
    def conversation_modes(self) -> tuple[Literal["cascade", "turn", "duplex"], ...]:
        if (
            not self.available
            or platform.system() != "Darwin"
            or platform.machine() != "arm64"
        ):
            return ()
        if isinstance(self, MiniCPMInfo):
            return ("duplex", "turn")
        if isinstance(self, (NemotronInfo, PersonaPlexInfo)):
            return ("duplex",)
        if isinstance(self, (QwenOmniInfo, MingAudioInfo)):
            return ("turn",)
        return ("cascade",)


class NemotronInfo(ModelInfo):
    id: Literal["nemotron"] = "nemotron"
    kind: Literal["duplex"] = "duplex"
    input_sample_rate: Literal[16000] = 16000
    output_sample_rate: Literal[22050] = 22050
    endpoint: Literal["/api/nemotron/session"] = "/api/nemotron/session"


class SesameInfo(ModelInfo):
    id: Literal["sesame"] = "sesame"
    kind: Literal["synthesis"] = "synthesis"
    output_sample_rate: Literal[24000] = 24000
    endpoint: Literal["/api/sesame/session"] = "/api/sesame/session"


class BreezeInfo(ModelInfo):
    id: Literal["breeze"] = "breeze"
    kind: Literal["synthesis"] = "synthesis"
    output_sample_rate: Literal[24000] = 24000
    endpoint: Literal["/api/breeze/session"] = "/api/breeze/session"


class PhononInfo(ModelInfo):
    id: Literal["phonon"] = "phonon"
    kind: Literal["synthesis"] = "synthesis"
    output_sample_rate: Literal[24000] = 24000
    endpoint: Literal["/api/phonon/session"] = "/api/phonon/session"


class KokoroInfo(ModelInfo):
    id: Literal["kokoro"] = "kokoro"
    kind: Literal["synthesis"] = "synthesis"
    output_sample_rate: Literal[24000] = 24000
    endpoint: Literal["/api/kokoro/session"] = "/api/kokoro/session"


class Audio8Info(ModelInfo):
    id: Literal["audio8"] = "audio8"
    kind: Literal["synthesis"] = "synthesis"
    output_sample_rate: Literal[44100] = 44100
    endpoint: Literal["/api/audio8/session"] = "/api/audio8/session"


class MingTTSInfo(ModelInfo):
    id: Literal["ming_tts"] = "ming_tts"
    kind: Literal["synthesis"] = "synthesis"
    output_sample_rate: Literal[44100] = 44100
    endpoint: Literal["/api/ming_tts/session"] = "/api/ming_tts/session"


class DramaInfo(ModelInfo):
    id: Literal["drama"] = "drama"
    kind: Literal["synthesis"] = "synthesis"
    output_sample_rate: Literal[44100] = 44100
    endpoint: Literal["/api/drama/session"] = "/api/drama/session"


class MayaInfo(ModelInfo):
    id: Literal["maya"] = "maya"
    kind: Literal["synthesis"] = "synthesis"
    output_sample_rate: Literal[24000] = 24000
    endpoint: Literal["/api/maya/session"] = "/api/maya/session"


class OmniVoiceInfo(ModelInfo):
    id: Literal["omnivoice"] = "omnivoice"
    kind: Literal["synthesis"] = "synthesis"
    output_sample_rate: Literal[24000] = 24000
    endpoint: Literal["/api/omnivoice/session"] = "/api/omnivoice/session"


class KittenInfo(ModelInfo):
    id: Literal["kitten"] = "kitten"
    kind: Literal["synthesis"] = "synthesis"
    output_sample_rate: Literal[24000] = 24000
    endpoint: Literal["/api/kitten/session"] = "/api/kitten/session"


class PersonaPlexInfo(ModelInfo):
    id: Literal["personaplex"] = "personaplex"
    kind: Literal["duplex"] = "duplex"
    output_sample_rate: Literal[24000] = 24000
    endpoint: Literal["/api/personaplex/session"] = "/api/personaplex/session"
    input_sample_rate: Literal[24000] = 24000


class MingAudioInfo(ModelInfo):
    id: Literal["ming_audio"] = "ming_audio"
    kind: Literal["conversation"] = "conversation"
    output_sample_rate: Literal[44100] = 44100
    endpoint: Literal["/api/ming_audio/session"] = "/api/ming_audio/session"


class QwenOmniInfo(ModelInfo):
    id: Literal["qwen_omni"] = "qwen_omni"
    kind: Literal["conversation"] = "conversation"
    output_sample_rate: Literal[24000] = 24000
    endpoint: Literal["/api/qwen_omni/session"] = "/api/qwen_omni/session"


class MiniCPMInfo(ModelInfo):
    id: Literal["minicpm"] = "minicpm"
    kind: Literal["duplex"] = "duplex"
    input_sample_rate: Literal[16000] = 16000
    output_sample_rate: Literal[24000] = 24000
    endpoint: Literal["/api/minicpm/session"] = "/api/minicpm/session"


ModelInfoUnion = Annotated[
    NemotronInfo
    | SesameInfo
    | BreezeInfo
    | PhononInfo
    | KokoroInfo
    | Audio8Info
    | MingTTSInfo
    | DramaInfo
    | MayaInfo
    | OmniVoiceInfo
    | KittenInfo
    | PersonaPlexInfo
    | MingAudioInfo
    | QwenOmniInfo
    | MiniCPMInfo,
    Field(discriminator="id"),
]
