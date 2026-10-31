"""Immutable Qwen speech runs, including partial audio and failed evidence."""

import hashlib
import wave
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from voicelab.runs.evidence import request_evidence
from voicelab_core.provenance import RuntimeEvidence, runtime_evidence
from voicelab_inference.qwen_omni.mlx.contracts import Metrics
from voicelab_inference.qwen_omni.mlx.runtime import QwenOmniMLX
from voicelab_models.qwen_omni.model import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    OUTPUT_SAMPLE_RATE,
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
    runtime: str = "qwen_omni/mlx"
    implementation: str = "mlx-vlm 0.7.11 primitives; lab-owned cancellable decoding"
    status: Literal["running", "completed", "incomplete", "failed"] = "running"
    request: dict[str, JsonValue]
    evidence: RuntimeEvidence
    load_seconds: float | None = None
    metrics: Metrics | None = None
    text: str = ""
    output_sha256: str | None = None
    error: str | None = None
    timing_note: str = (
        "Output RTF includes preprocessing, Thinker, Talker, codec and delivery "
        "backpressure. Loading is separate. First PCM is measured before delivery."
    )


def execute_run(
    request: Request, output: Path, runtime: QwenOmniMLX | None = None
) -> Path:
    backend = runtime or QwenOmniMLX()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "run.json"
    audio_path = output / "response.wav"
    report = Report(
        request=TypeAdapter(dict[str, JsonValue]).validate_python(
            request_evidence(request)
        ),
        evidence=runtime_evidence(
            Path.cwd(), ("mlx", "mlx-vlm", "transformers", "numpy")
        ),
    )
    path.write_text(report.model_dump_json(indent=2) + "\n")
    text: list[str] = []
    try:
        backend.load()
        report = report.model_copy(update={"load_seconds": backend.load_seconds})
        if request.return_audio:
            with wave.open(str(audio_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(OUTPUT_SAMPLE_RATE)
                response = backend.respond(request, text.append, audio.writeframes)
        else:
            response = backend.respond(request, text.append)
        report = report.model_copy(
            update={
                "status": "completed" if response.metrics.completed else "incomplete",
                "text": response.text,
                "metrics": response.metrics,
            }
        )
    except BaseException as exc:
        report = report.model_copy(
            update={"status": "failed", "error": str(exc), "text": "".join(text)}
        )
        raise
    finally:
        try:
            backend.close()
        finally:
            (output / "response.txt").write_text(report.text + "\n")
            if audio_path.exists():
                report = report.model_copy(
                    update={
                        "output_sha256": hashlib.sha256(
                            audio_path.read_bytes()
                        ).hexdigest()
                    }
                )
            path.write_text(report.model_dump_json(indent=2) + "\n")
    if report.status == "incomplete":
        raise RuntimeError(f"Qwen reached a text or speech generation cap; see {path}.")
    return path
