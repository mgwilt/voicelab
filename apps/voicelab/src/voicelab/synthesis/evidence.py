"""Describe request inputs without copying reference recordings into reports."""

import hashlib

from pydantic import BaseModel

from voicelab_models.audio import ReferenceAudio


def request_evidence(value: object) -> object:
    if isinstance(value, ReferenceAudio):
        return {
            "sha256": hashlib.sha256(value.wav_bytes()).hexdigest(),
            "sample_rate": value.sample_rate,
            "duration_seconds": value.duration_seconds,
        }
    if isinstance(value, BaseModel):
        return {
            name: request_evidence(getattr(value, name))
            for name in type(value).model_fields
        }
    if isinstance(value, (tuple, list)):
        return [request_evidence(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported request evidence: {type(value).__name__}")
