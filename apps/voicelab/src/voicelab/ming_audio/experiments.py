"""Retain turn-based Ming Audio request evidence and failures."""

import base64
import hashlib
import wave
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from voicelab_inference.ming_audio.mlx.runtime import (
    Metrics,
    MingAudioMLX,
)
from voicelab_models.ming_audio.model import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    OUTPUT_SAMPLE_RATE,
    REVISION,
    SOURCE_MODEL,
    SOURCE_REVISION,
    Request,
    Task,
)


class HistoryEvidence(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    role: Literal["user", "assistant"]
    content: str
    audio_sha256: str | None


class Report(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    schema_version: int = 1
    model: str = SOURCE_MODEL
    source_checkpoint_revision: str = REVISION
    implementation_source_revision: str = SOURCE_REVISION
    checkpoint_id: str = CHECKPOINT_ID
    checkpoint_revision: str = CHECKPOINT_REVISION
    load_seconds: float | None = None
    runtime: str = "ming_audio/mlx"
    status: Literal["running", "completed", "incomplete", "failed"] = "running"
    input_sha256: str | None
    reference_sha256: str | None = None
    reference_text: str | None = None
    trim_reference_silence: bool = True
    task: Task = "spoken_dialogue"
    system_prompt: str | None = None
    history: tuple[HistoryEvidence, ...] = ()
    thinking: bool = False
    reasoning: str | None = None
    language: str | None = None
    prompt: str
    voice: str
    max_tokens: int
    seed: int = 0
    max_decode_steps: int = 375
    cfg_scale: float = 2.0
    sigma: float = 0.25
    temperature: float = 0.0
    flow_steps: int = 10
    text: str | None = None
    metrics: Metrics | None = None
    output_sha256: str | None = None
    error: str | None = None
    timing_note: str = (
        "Local buffered Thinker/Talker response. Output RTF includes inference "
        "and decoding; model loading is measured separately."
    )


def run_experiment(request: Request, output: Path) -> Path:
    backend = MingAudioMLX()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "run.json"
    report = Report(
        input_sha256=hashlib.sha256(
            base64.b64decode(request.audio_wav_base64)
        ).hexdigest()
        if request.audio_wav_base64 is not None
        else None,
        reference_sha256=hashlib.sha256(request.reference_audio.wav_bytes()).hexdigest()
        if request.reference_audio is not None
        else None,
        reference_text=request.reference_text,
        trim_reference_silence=request.trim_reference_silence,
        task=request.task,
        system_prompt=request.system_prompt,
        history=tuple(
            HistoryEvidence(
                role=turn.role,
                content=turn.content,
                audio_sha256=hashlib.sha256(turn.audio.wav_bytes()).hexdigest()
                if turn.audio is not None
                else None,
            )
            for turn in request.history
        ),
        thinking=request.thinking,
        prompt=request.prompt,
        voice=request.voice,
        max_tokens=request.max_tokens,
        seed=request.seed,
        max_decode_steps=request.max_decode_steps,
        cfg_scale=request.cfg_scale,
        sigma=request.sigma,
        temperature=request.temperature,
        flow_steps=request.flow_steps,
    )
    report_path.write_text(report.model_dump_json(indent=2) + "\n")
    try:
        backend.load()
        report = report.model_copy(update={"load_seconds": backend.load_seconds})
        response = backend.respond(request)
        audio_path = output / "response.wav"
        if response.pcm:
            with wave.open(str(audio_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(OUTPUT_SAMPLE_RATE)
                audio.writeframes(response.pcm)
        (output / "response.txt").write_text(response.text + "\n")
        report = report.model_copy(
            update={
                "status": "completed" if response.metrics.completed else "incomplete",
                "metrics": response.metrics,
                "text": response.text,
                "output_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest()
                if response.pcm
                else None,
                "reasoning": response.reasoning,
                "language": response.language,
            }
        )
    except BaseException as exc:
        report = report.model_copy(update={"status": "failed", "error": str(exc)})
        raise
    finally:
        try:
            backend.close()
        finally:
            report_path.write_text(report.model_dump_json(indent=2) + "\n")
    if report.status == "incomplete":
        raise RuntimeError(
            f"Ming Audio reached a text or speech generation limit; see {report_path}."
        )
    return report_path
