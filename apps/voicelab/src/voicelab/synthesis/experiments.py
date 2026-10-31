"""Retain audio, failures, and provenance for typed synthesis experiments."""

import hashlib
import json
import wave
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from voicelab.synthesis.evidence import request_evidence
from voicelab_core.provenance import runtime_evidence
from voicelab_core.synthesis import SynthesisMetrics

type Audio = NDArray[np.float32]


class SynthesisBackend[Request: BaseModel](Protocol):
    @property
    def implementation(self) -> str: ...
    @property
    def precision(self) -> str: ...
    @property
    def device(self) -> str: ...

    def load(self) -> None: ...
    def close(self) -> None: ...
    def synthesize(
        self, request: Request, emit: Callable[[Audio], None]
    ) -> SynthesisMetrics: ...


@dataclass(frozen=True)
class AuxiliaryCheckpoint:
    role: str
    checkpoint: str
    revision: str
    filename: str | None = None


@dataclass(frozen=True)
class ModelEvidence:
    model: str
    revision: str
    checkpoint: str
    checkpoint_revision: str
    sample_rate: int
    dependencies: tuple[str, ...]
    auxiliary_checkpoints: tuple[AuxiliaryCheckpoint, ...] = ()


def run_trial[Request: BaseModel](
    runtime: SynthesisBackend[Request],
    request: Request,
    model: ModelEvidence,
    output: Path,
    workspace: Path,
    *,
    warmup: bool = False,
) -> SynthesisMetrics:
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, object] = {
        **asdict(model),
        "implementation": runtime.implementation,
        "device": runtime.device,
        "precision": runtime.precision,
        "request": request_evidence(request),
        "warmup": warmup,
        "timing": (
            "Load excluded. Generation includes decoded audio delivery and WAV writes."
        ),
    }
    path = output / "speech.wav"
    try:
        report["runtime"] = asdict(runtime_evidence(workspace, model.dependencies))
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(model.sample_rate)

            def emit(samples: Audio) -> None:
                if (
                    samples.ndim != 1
                    or not samples.size
                    or not np.isfinite(samples).all()
                ):
                    raise ValueError("Expected finite, non-empty mono audio")
                pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
                audio.writeframes(pcm.tobytes())

            metrics = runtime.synthesize(request, emit)
        report["metrics"] = asdict(metrics)
        report["status"] = metrics.completion_state
        if metrics.completion_state == "incomplete":
            raise RuntimeError(
                "Generation ended without completion; partial audio retained"
            )
        return metrics
    except BaseException as exc:
        report.setdefault(
            "status", "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        report["error"] = str(exc)
        raise
    finally:
        if path.exists():
            report["output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
