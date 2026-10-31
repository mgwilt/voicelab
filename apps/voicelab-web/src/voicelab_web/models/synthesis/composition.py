"""Explicit composition of additional synthesis models; imports stay lazy."""

from voicelab_models.audio8.model import SAMPLE_RATE as Audio8MLXRate
from voicelab_models.audio8.model import Request as Audio8MLXRequest
from voicelab_models.drama.model import SAMPLE_RATE as DramaMLXRate
from voicelab_models.drama.model import Request as DramaMLXRequest
from voicelab_models.kitten.model import SAMPLE_RATE as KittenTorchRate
from voicelab_models.kitten.model import Request as KittenTorchRequest
from voicelab_models.kokoro.model import SAMPLE_RATE as KokoroMLXRate
from voicelab_models.kokoro.model import Request as KokoroMLXRequest
from voicelab_models.maya.model import SAMPLE_RATE as MayaMLXRate
from voicelab_models.maya.model import Request as MayaMLXRequest
from voicelab_models.ming_tts.model import SAMPLE_RATE as MingTTSMLXRate
from voicelab_models.ming_tts.model import Request as MingTTSMLXRequest
from voicelab_models.omnivoice.model import SAMPLE_RATE as OmniVoiceTorchRate
from voicelab_models.omnivoice.model import Request as OmniVoiceTorchRequest
from voicelab_models.phonon.model import SAMPLE_RATE as PhononTorchRate
from voicelab_models.phonon.model import Request as PhononTorchRequest
from voicelab_web.models.synthesis.routes import SynthesisRoutes
from voicelab_web.models.synthesis.service import SynthesisService, Synthesizer
from voicelab_web.schemas import ServerConfig


def install(routes: SynthesisRoutes, config: ServerConfig) -> None:
    def phonon() -> Synthesizer[PhononTorchRequest]:
        from voicelab_inference.phonon.torch.runtime import PhononTorch

        return PhononTorch(config.phonon_device)

    routes.add(
        "phonon",
        SynthesisService(PhononTorchRequest, phonon, sample_rate=PhononTorchRate),
    )

    def kokoro() -> Synthesizer[KokoroMLXRequest]:
        from voicelab_inference.kokoro.mlx.runtime import KokoroMLX

        return KokoroMLX()

    routes.add(
        "kokoro", SynthesisService(KokoroMLXRequest, kokoro, sample_rate=KokoroMLXRate)
    )

    def audio8() -> Synthesizer[Audio8MLXRequest]:
        from voicelab_inference.audio8.mlx.runtime import Audio8MLX

        return Audio8MLX()

    routes.add(
        "audio8", SynthesisService(Audio8MLXRequest, audio8, sample_rate=Audio8MLXRate)
    )

    def ming_tts() -> Synthesizer[MingTTSMLXRequest]:
        from voicelab_inference.ming_tts.mlx.runtime import MingTTSMLX

        return MingTTSMLX()

    routes.add(
        "ming_tts",
        SynthesisService(MingTTSMLXRequest, ming_tts, sample_rate=MingTTSMLXRate),
    )

    def drama() -> Synthesizer[DramaMLXRequest]:
        from voicelab_inference.drama.mlx.runtime import DramaMLX

        return DramaMLX()

    routes.add(
        "drama", SynthesisService(DramaMLXRequest, drama, sample_rate=DramaMLXRate)
    )

    def maya() -> Synthesizer[MayaMLXRequest]:
        from voicelab_inference.maya.mlx.runtime import MayaMLX

        return MayaMLX()

    routes.add("maya", SynthesisService(MayaMLXRequest, maya, sample_rate=MayaMLXRate))

    def omnivoice() -> Synthesizer[OmniVoiceTorchRequest]:
        from voicelab_inference.omnivoice.torch.runtime import OmniVoiceTorch

        return OmniVoiceTorch(config.omnivoice_device)

    routes.add(
        "omnivoice",
        SynthesisService(
            OmniVoiceTorchRequest, omnivoice, sample_rate=OmniVoiceTorchRate
        ),
    )

    def kitten() -> Synthesizer[KittenTorchRequest]:
        from voicelab_inference.kitten.torch.runtime import KittenTorch

        return KittenTorch(config.kitten_device)

    routes.add(
        "kitten",
        SynthesisService(KittenTorchRequest, kitten, sample_rate=KittenTorchRate),
    )
