"""Explicit model capabilities presented by the lab, not a plugin registry."""

import platform
import sys

from voicelab_web.catalog_types import (
    Audio8Info,
    BreezeInfo,
    DramaInfo,
    KittenInfo,
    KokoroInfo,
    MayaInfo,
    MingAudioInfo,
    MingTTSInfo,
    NemotronInfo,
    OmniVoiceInfo,
    PersonaPlexInfo,
    PhononInfo,
    SesameInfo,
)
from voicelab_web.catalog_types import ModelId as ModelId
from voicelab_web.catalog_types import ModelInfoUnion as ModelInfoUnion
from voicelab_web.schemas import ServerConfig


def models(config: ServerConfig) -> tuple[ModelInfoUnion, ...]:
    apple = sys.platform == "darwin" and platform.machine() == "arm64"
    return (
        NemotronInfo(
            label="Nemotron VoiceChat 11B",
            description="Speak naturally. Listen and talk at the same time.",
            backend="MLX · 8-bit",
            available=apple,
            reason=None if apple else "This backend requires Apple Silicon and MLX.",
        ),
        SesameInfo(
            label="Sesame CSM-1B",
            description="Turn text into speech with contextual voice synthesis.",
            backend=f"PyTorch · {config.sesame_device.upper()}",
            available=True,
        ),
        BreezeInfo(
            label="Breeze TTS",
            description="Design, clone, and direct a voice.",
            backend="MLX",
            available=apple,
            reason=None if apple else "This backend requires Apple Silicon and MLX.",
        ),
        PhononInfo(
            label="Phonon 31M",
            description="Streaming English speech with preset voices.",
            backend="PyTorch",
            available=True,
            reason=None,
        ),
        KokoroInfo(
            label="Kokoro 82M",
            description="Multilingual speech with preset voices and blends.",
            backend="MLX",
            available=apple,
            reason=None if apple else "This runtime requires Apple Silicon and MLX.",
        ),
        Audio8Info(
            label="Audio8 Realistic 0.6B",
            description="Multilingual speech and reference voice conditioning.",
            backend="MLX",
            available=apple,
            reason=None if apple else "This runtime requires Apple Silicon and MLX.",
        ),
        MingTTSInfo(
            label="Ming 3.0 TTS 0.5B",
            description="Instruction-guided multilingual speech.",
            backend="MLX",
            available=apple,
            reason=None if apple else "This runtime requires Apple Silicon and MLX.",
        ),
        DramaInfo(
            label="Fish Audio Drama 3",
            description="Expressive speech with multiple speaker tags.",
            backend="MLX",
            available=apple,
            reason=None if apple else "This runtime requires Apple Silicon and MLX.",
        ),
        MayaInfo(
            label="Maya 2 Native",
            description="Multilingual English and Indic speech.",
            backend="MLX",
            available=apple,
            reason=None if apple else "This runtime requires Apple Silicon and MLX.",
        ),
        OmniVoiceInfo(
            label="OmniVoice",
            description="Diffusion speech synthesis with voice instructions.",
            backend="PyTorch",
            available=True,
            reason=None,
        ),
        KittenInfo(
            label="KittenTTS 2",
            description="Speech with preset voices and expression controls.",
            backend="PyTorch",
            available=True,
            reason=None,
        ),
        PersonaPlexInfo(
            label="PersonaPlex 7B",
            description="Listen and speak at the same time.",
            backend="MLX",
            available=apple,
            reason=None if apple else "This runtime requires Apple Silicon and MLX.",
        ),
        MingAudioInfo(
            label="Ming 3.0 Audio Tiny",
            description="Speech, transcription, audio understanding, and text chat.",
            backend="MLX",
            available=apple,
            reason=None if apple else "This runtime requires Apple Silicon and MLX.",
        ),
    )
