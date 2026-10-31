"""Discriminated conversation requests reuse validated model controls."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from voicelab_models.audio8.model import Request as Audio8Settings
from voicelab_models.breeze.synthesis import SynthesisRequest as BreezeSettings
from voicelab_models.drama.model import Request as DramaSettings
from voicelab_models.kitten.model import Request as KittenSettings
from voicelab_models.kokoro.model import Request as KokoroSettings
from voicelab_models.maya.model import Request as MayaSettings
from voicelab_models.ming_audio.model import Request as MingAudioSettings
from voicelab_models.ming_tts.model import Request as MingTtsSettings
from voicelab_models.minicpm.model import Request as MinicpmSettings
from voicelab_models.omnivoice.model import Request as OmnivoiceSettings
from voicelab_models.personaplex.model import Request as PersonaplexSettings
from voicelab_models.phonon.model import Request as PhononSettings
from voicelab_models.qwen_omni.model import Request as QwenOmniSettings
from voicelab_models.sesame.model import SynthesisRequest as SesameSettings
from voicelab_web.models.nemotron.schemas import StartMessage as NemotronSettings


class Common(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    system_prompt: str = Field(
        default="Answer briefly and naturally. Use plain spoken language.",
        min_length=1,
        max_length=2000,
    )


class SesameConversation(Common):
    model: Literal["sesame"]
    mode: Literal["cascade"] = "cascade"
    settings: SesameSettings = Field(
        default_factory=lambda: SesameSettings(text="Hello.")
    )


class BreezeConversation(Common):
    model: Literal["breeze"]
    mode: Literal["cascade"] = "cascade"
    settings: BreezeSettings = Field(
        default_factory=lambda: BreezeSettings(text="Hello.")
    )


class PhononConversation(Common):
    model: Literal["phonon"]
    mode: Literal["cascade"] = "cascade"
    settings: PhononSettings = Field(
        default_factory=lambda: PhononSettings(text="Hello.")
    )


class KokoroConversation(Common):
    model: Literal["kokoro"]
    mode: Literal["cascade"] = "cascade"
    settings: KokoroSettings = Field(
        default_factory=lambda: KokoroSettings(text="Hello.")
    )


class Audio8Conversation(Common):
    model: Literal["audio8"]
    mode: Literal["cascade"] = "cascade"
    settings: Audio8Settings = Field(
        default_factory=lambda: Audio8Settings(text="Hello.")
    )


class MingTtsConversation(Common):
    model: Literal["ming_tts"]
    mode: Literal["cascade"] = "cascade"
    settings: MingTtsSettings = Field(
        default_factory=lambda: MingTtsSettings(text="Hello.")
    )


class DramaConversation(Common):
    model: Literal["drama"]
    mode: Literal["cascade"] = "cascade"
    settings: DramaSettings = Field(
        default_factory=lambda: DramaSettings(text="Hello.")
    )


class MayaConversation(Common):
    model: Literal["maya"]
    mode: Literal["cascade"] = "cascade"
    settings: MayaSettings = Field(default_factory=lambda: MayaSettings(text="Hello."))


class OmnivoiceConversation(Common):
    model: Literal["omnivoice"]
    mode: Literal["cascade"] = "cascade"
    settings: OmnivoiceSettings = Field(
        default_factory=lambda: OmnivoiceSettings(text="Hello.")
    )


class KittenConversation(Common):
    model: Literal["kitten"]
    mode: Literal["cascade"] = "cascade"
    settings: KittenSettings = Field(
        default_factory=lambda: KittenSettings(text="Hello.")
    )


class QwenOmniConversation(Common):
    model: Literal["qwen_omni"]
    mode: Literal["turn"] = "turn"
    settings: QwenOmniSettings = Field(default_factory=QwenOmniSettings)


class MingAudioConversation(Common):
    model: Literal["ming_audio"]
    mode: Literal["turn"] = "turn"
    settings: MingAudioSettings = Field(default_factory=MingAudioSettings)


class PersonaplexConversation(Common):
    model: Literal["personaplex"]
    mode: Literal["duplex"] = "duplex"
    settings: PersonaplexSettings = Field(default_factory=PersonaplexSettings)


class MinicpmConversation(Common):
    model: Literal["minicpm"]
    mode: Literal["turn", "duplex"] = "duplex"
    settings: MinicpmSettings = Field(default_factory=MinicpmSettings)


class NemotronConversation(Common):
    model: Literal["nemotron"]
    mode: Literal["duplex"] = "duplex"
    settings: NemotronSettings = Field(
        default_factory=lambda: NemotronSettings(type="start")
    )


ConversationRequest = Annotated[
    SesameConversation
    | BreezeConversation
    | PhononConversation
    | KokoroConversation
    | Audio8Conversation
    | MingTtsConversation
    | DramaConversation
    | MayaConversation
    | OmnivoiceConversation
    | KittenConversation
    | QwenOmniConversation
    | MingAudioConversation
    | PersonaplexConversation
    | MinicpmConversation
    | NemotronConversation,
    Field(discriminator="model"),
]
REQUEST = TypeAdapter(ConversationRequest)
