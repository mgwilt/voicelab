"""Local, immutable synthesis runs with partial audio and failure evidence."""

import hashlib
import json
import time
import wave
from dataclasses import asdict
from pathlib import Path

from voicelab.runs.evidence import request_evidence
from voicelab_core.provenance import runtime_evidence
from voicelab_core.synthesis import SynthesisMetrics
from voicelab_inference.sesame.torch.audio import Audio, pcm16
from voicelab_inference.sesame.torch.runtime import ReferenceSpeech, ResidentCSM
from voicelab_models.sesame.model import (
    MODEL_ID,
    REVISION,
    SAMPLE_RATE,
    SynthesisRequest,
)


def run_trial(
    runtime: ResidentCSM,
    request: SynthesisRequest,
    output: Path,
    workspace: Path,
    reference: ReferenceSpeech | None = None,
    *,
    warmup: bool = False,
) -> SynthesisMetrics:
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, object] = {
        "model": MODEL_ID,
        "revision": REVISION,
        "implementation": f"csm-torch/{runtime.engine}",
        "device": runtime.device,
        "precision": runtime.precision,
        "codec_precision": "float32",
        "request": request_evidence(request),
        "warmup": warmup,
        "runtime": asdict(
            runtime_evidence(workspace, ("torch", "transformers", "numpy"))
        ),
        "runtime_lock_sha256": hashlib.sha256(
            (workspace / "uv.lock").read_bytes()
        ).hexdigest(),
        "reference_sha256": hashlib.sha256(reference.samples.tobytes()).hexdigest()
        if reference
        else None,
        "reference_transcript": reference.transcript if reference else None,
        "timing": (
            "Generation starts after tokenization; includes codec and WAV writes. "
            "Load excluded."
        ),
    }
    started = time.perf_counter()
    path = output / "speech.wav"
    try:
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(SAMPLE_RATE)

            def emit(samples: Audio) -> None:
                audio.writeframes(pcm16(samples))

            metrics = runtime.synthesize(request, emit, reference)
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
        report["wall_seconds"] = time.perf_counter() - started
        if path.exists():
            report["output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
