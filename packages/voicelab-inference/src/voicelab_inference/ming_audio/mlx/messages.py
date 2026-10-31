"""Translate validated conversation turns into the pinned model's message format."""

from typing import Literal, TypedDict

import mlx.core as mx
import numpy as np
from mlx_audio.stt.models.ming_v3_asr.ming_v3_asr import ASR_INSTRUCTION

from voicelab_inference.audio import reference_pcm
from voicelab_models.ming_audio.model import Request, decode_input


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class AudioContent(TypedDict):
    type: Literal["audio"]
    audio: mx.array


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str | list[TextContent | AudioContent]


def messages(request: Request) -> list[Message]:
    result: list[Message] = []
    if request.system_prompt is not None:
        result.append({"role": "system", "content": request.system_prompt})
    for turn in request.history:
        if turn.audio is None:
            result.append({"role": turn.role, "content": turn.content})
        else:
            result.append(
                {
                    "role": turn.role,
                    "content": [
                        {
                            "type": "audio",
                            "audio": mx.array(reference_pcm(turn.audio, 16000)),
                        },
                        {"type": "text", "text": turn.content},
                    ],
                }
            )
    prompt = request.prompt
    if request.task == "asr" and prompt == "Please answer the question in the audio.":
        prompt = ASR_INSTRUCTION
    content: list[TextContent | AudioContent] = []
    if request.audio_wav_base64 is not None:
        audio = decode_input(request.audio_wav_base64)
        waveform = np.frombuffer(audio.pcm, dtype="<i2").astype(np.float32) / 32768
        content.append({"type": "audio", "audio": mx.array(waveform)})
    content.append({"type": "text", "text": prompt})
    result.append({"role": "user", "content": content})
    return result
