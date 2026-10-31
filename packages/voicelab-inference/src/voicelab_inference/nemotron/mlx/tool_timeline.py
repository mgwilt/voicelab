"""Prepare original and source-faithful expanded Nemotron audio timelines."""

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx_audio.stt.models.nemotron_asr.audio import log_mel_spectrogram
from mlx_audio.stt.utils import load_audio
from mlx_vlm.models.nemotron_voicechat.session import VoiceChatSession

from voicelab_models.nemotron.tools import ToolTimeline


@dataclass(frozen=True)
class AudioTimeline:
    audio: mx.array
    asr: mx.array
    prompt_frames: int
    original_frames: int


def prepare(
    session: VoiceChatSession, source: Path, prompt: str, trailing_seconds: float
) -> AudioTimeline:
    model = session.model
    waveform = load_audio(str(source), sr=model.config.input_sample_rate).squeeze()
    waveform = mx.pad(
        waveform, (0, round(trailing_seconds * model.config.input_sample_rate))
    )
    mel = log_mel_spectrogram(waveform, model.config.audio_config.preprocessor)
    audio, lengths, asr = model.stt_model.perception(
        mel, mx.array([mel.shape[1]], dtype=mx.int32)
    )
    frames = int(lengths[0])
    ids = [
        model.config.bos_token_id,
        *session.tokenizer.encode(prompt, add_special_tokens=False),
        model.config.eos_token_id,
    ]
    if len(ids) > 2048 or frames > 750:
        raise ValueError("Tool experiments support 2048 prompt tokens and 60 seconds.")
    embeds = model.stt_model.embed_tokens(mx.array([ids], dtype=mx.int32))
    audio = mx.concatenate([embeds.astype(audio.dtype), audio[:, :frames]], axis=1)
    mx.eval(audio, asr)
    return AudioTimeline(audio, asr[:, :frames], len(ids), frames)


def expand(
    session: VoiceChatSession, original: AudioTimeline, plan: ToolTimeline
) -> AudioTimeline:
    """Each inserted segment starts at the beginning of the 60-second template."""
    model = session.model
    if plan.call_start >= original.original_frames:
        raise ValueError("Tool insertion lies outside the original audio timeline.")
    silence = mx.zeros((60 * model.config.input_sample_rate,), dtype=mx.float32)
    mel = log_mel_spectrogram(silence, model.config.audio_config.preprocessor)
    template, lengths, _ = model.stt_model.perception(
        mel, mx.array([mel.shape[1]], dtype=mx.int32)
    )
    template = template[:, : int(lengths[0])].astype(original.audio.dtype)
    parts: list[mx.array] = []
    for count in (plan.call_tokens + 2, len(plan.response_tokens), 1):
        if count > template.shape[1]:
            raise ValueError("Inserted tool segment exceeds the silence template.")
        parts.append(template[:, :count])
    anchor = original.prompt_frames + plan.call_start
    audio = mx.concatenate(
        [original.audio[:, :anchor], *parts, original.audio[:, anchor:]], axis=1
    )
    mx.eval(audio)
    return AudioTimeline(
        audio, original.asr, original.prompt_frames, original.original_frames
    )
