"""Immutable Breeze synthesis evidence, including partial and failed runs."""

import hashlib
import json
import wave
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from voicelab.runs.evidence import request_evidence
from voicelab_core.provenance import runtime_evidence
from voicelab_core.synthesis import SynthesisMetrics
from voicelab_inference.breeze.mlx.audio import Audio, pcm16
from voicelab_models.breeze.model import (
    MLX_MODEL_ID,
    MLX_REVISION,
    MODEL_ID,
    REVISION,
    SAMPLE_RATE,
)
from voicelab_models.breeze.synthesis import SynthesisRequest


class SynthesisBackend(Protocol):
    implementation: str
    precision: str
    device: str

    def synthesize(
        self, request: SynthesisRequest, emit: Callable[[Audio], None]
    ) -> SynthesisMetrics: ...


def run_trial(
    runtime: SynthesisBackend,
    request: SynthesisRequest,
    output: Path,
    workspace: Path,
    *,
    warmup: bool = False,
) -> SynthesisMetrics:
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, object] = {
        "model": MODEL_ID,
        "revision": REVISION,
        "checkpoint": MLX_MODEL_ID,
        "checkpoint_revision": MLX_REVISION,
        "implementation": runtime.implementation,
        "device": runtime.device,
        "precision": runtime.precision,
        "request": request_evidence(request),
        "warmup": warmup,
        "runtime": asdict(runtime_evidence(workspace, ("mlx", "mlx-audio", "numpy"))),
        "timing": (
            "Load excluded; prefill, sampling, codec, synchronization "
            "and WAV writes included."
        ),
    }
    path = output / "speech.wav"
    try:
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(SAMPLE_RATE)

            def emit(samples: Audio) -> None:
                audio.writeframes(pcm16(samples))

            metrics = runtime.synthesize(request, emit)
        report["metrics"] = asdict(metrics)
        report["status"] = metrics.completion_state
        if metrics.completion_state != "complete":
            raise RuntimeError("Frame limit reached before EOS; partial audio retained")
        return metrics
    except Exception as exc:
        report.setdefault("status", "failed")
        report["error"] = str(exc)
        raise
    finally:
        if path.exists():
            report["output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
