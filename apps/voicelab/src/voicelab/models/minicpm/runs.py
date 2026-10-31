"""Retain audio and provenance for successful, capped, and failed runs."""

import hashlib
import math
import wave
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from voicelab.runs.evidence import request_evidence
from voicelab_inference.minicpm.native.prepare import patch_hash
from voicelab_inference.minicpm.native.protocol import Event
from voicelab_inference.minicpm.native.runtime import Metrics, MiniCPMNative
from voicelab_models.minicpm.model import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    FRAME_BYTES,
    INPUT_RATE,
    MODEL_FILE,
    NATIVE_REVISION,
    OUTPUT_RATE,
    SOURCE_MODEL,
    SOURCE_REVISION,
    Request,
)


class Report(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    schema_version: int = 1
    model: str = SOURCE_MODEL
    source_revision: str = SOURCE_REVISION
    checkpoint: str = CHECKPOINT_ID
    checkpoint_revision: str = CHECKPOINT_REVISION
    checkpoint_file: str = MODEL_FILE
    implementation_revision: str = NATIVE_REVISION
    patch_sha256: str
    runtime: str = "minicpm/llama.cpp-omni"
    precision: str = MiniCPMNative.precision
    device: str = MiniCPMNative.device
    status: Literal["running", "complete", "incomplete", "stopped", "failed"] = (
        "running"
    )
    request: JsonValue
    input_sha256: str | None
    reference_sha256: str | None
    trailing_silence_seconds: float
    setup_seconds: float | None = None
    metrics: Metrics | None = None
    text: str = ""
    error: str | None = None
    timing_note: str = (
        "RTF divides native processing and delivery time by input duration for duplex "
        "or output duration for turns. Excludes preparation and input pacing. "
        "Duplex capture stop is not utterance EOS."
    )


def read_duplex(source: Path, silence: float) -> bytes:
    if not math.isfinite(silence) or not 0 <= silence <= 10:
        raise ValueError("Trailing silence must be between zero and ten seconds.")
    with wave.open(str(source), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (
            1,
            2,
            INPUT_RATE,
        ):
            raise ValueError("Duplex input must be mono PCM16 WAV at 16 kHz.")
        count = audio.getnframes()
        if not 0 < count <= (120 - silence) * INPUT_RATE:
            raise ValueError("Input and silence must fit within 120 seconds.")
        pcm = audio.readframes(count)
        if len(pcm) != count * 2:
            raise ValueError("Input WAV is truncated.")
    pcm += bytes(int(silence * INPUT_RATE) * 2)
    return pcm + bytes((-len(pcm)) % FRAME_BYTES)


def execute_run(
    output: Path, request: Request, source: Path | None, silence: float
) -> Path:
    pcm = b""
    if request.mode == "full_duplex":
        if source is None:
            raise ValueError("Duplex replay needs an input WAV.")
        pcm = read_duplex(source, silence)
    backend = MiniCPMNative()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "run.json"
    report = Report(
        request=TypeAdapter(JsonValue).validate_python(
            request_evidence(request), strict=True
        ),
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest()
        if source
        else None,
        reference_sha256=hashlib.sha256(request.reference_audio.wav_bytes()).hexdigest()
        if request.reference_audio
        else None,
        trailing_silence_seconds=silence if request.mode == "full_duplex" else 0,
        patch_sha256=patch_hash(),
    )
    report_path.write_text(report.model_dump_json(indent=2) + "\n")
    text: list[str] = []
    try:
        with wave.open(str(output / "response.wav"), "wb") as audio:
            audio.setparams((1, 2, OUTPUT_RATE, 0, "NONE", ""))

            def emit(event: Event) -> None:
                if event.kind == "audio":
                    audio.writeframesraw(event.pcm)
                elif event.kind == "text":
                    text.append(event.text)

            backend.start(request)
            if request.mode == "full_duplex":
                for offset in range(0, len(pcm), FRAME_BYTES):
                    backend.push(pcm[offset : offset + FRAME_BYTES], emit)
            else:
                backend.turn(emit)
            backend.finish(emit)
        if request.mode == "full_duplex" and backend.metrics().output_seconds == 0:
            raise RuntimeError("MiniCPM produced no speech before capture ended.")
        status = (
            "stopped"
            if request.mode == "full_duplex"
            else ("complete" if backend.metrics().completed is True else "incomplete")
        )
        report = report.model_copy(update={"status": status})
    except BaseException as exc:
        report = report.model_copy(update={"status": "failed", "error": str(exc)})
        raise
    finally:
        report = report.model_copy(
            update={
                "setup_seconds": backend.setup_seconds,
                "metrics": backend.metrics(),
                "text": "".join(text),
            }
        )
        try:
            backend.close()
        finally:
            report_path.write_text(report.model_dump_json(indent=2) + "\n")
    return report_path
