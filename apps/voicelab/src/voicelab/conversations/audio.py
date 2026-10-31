"""Encode bounded microphone turns for existing model request boundaries."""

import base64
import io
import wave

from voicelab_models.audio import ReferenceAudio


def reference(pcm: bytes, rate: int = 16000) -> ReferenceAudio:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(pcm)
    return ReferenceAudio(
        wav_base64=base64.b64encode(buffer.getvalue()).decode("ascii")
    )
