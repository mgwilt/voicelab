"""Explicit Sesame synthesis and repeated local runs."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter

from voicelab.models.sesame.context import load_context
from voicelab.models.sesame.runs import run_trial
from voicelab_inference.sesame.torch.audio import read_reference
from voicelab_inference.sesame.torch.runtime import ReferenceSpeech, ResidentCSM
from voicelab_models.sesame.model import Device, Engine, SynthesisRequest


class Arguments(argparse.Namespace):
    device: str
    engine: str
    text: str
    output: Path
    workspace: Path
    speaker: int
    seed: int
    max_frames: int
    trials: int
    warmup: bool
    reference_audio: Path | None
    reference_text: str | None
    temperature: float
    top_k: int
    context: Path | None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sesame CSM synthesis and RTF measurements"
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    parser.add_argument(
        "--engine", choices=("baseline", "direct-depth"), default="direct-depth"
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--speaker", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=250)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--reference-text")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--context", type=Path, help="JSON list of prior audio/text/speaker turns"
    )
    args = parser.parse_args(namespace=Arguments())
    if not 1 <= args.trials <= 100:
        parser.error("trials must be between 1 and 100")
    if (args.reference_audio is None) != (args.reference_text is None):
        parser.error("Reference audio and transcript must be supplied together")
    if args.context is not None and args.reference_audio is not None:
        parser.error("Use --context or --reference-audio, not both")
    request = SynthesisRequest(
        text=args.text,
        speaker=TypeAdapter(Literal[0, 1]).validate_python(args.speaker),
        seed=args.seed,
        max_frames=args.max_frames,
        temperature=args.temperature,
        top_k=args.top_k,
        context=load_context(args.context) if args.context is not None else (),
    )
    reference = (
        ReferenceSpeech(read_reference(args.reference_audio), args.reference_text)
        if args.reference_audio is not None and args.reference_text is not None
        else None
    )
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        runtime = ResidentCSM(
            TypeAdapter(Device).validate_python(args.device),
            TypeAdapter(Engine).validate_python(args.engine),
        )
    except Exception as exc:
        (args.output / "load.json").write_text(
            json.dumps({"status": "failed", "error": str(exc)})
        )
        raise
    (args.output / "load.json").write_text(
        json.dumps(
            {
                "status": "loaded",
                "load_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n"
    )
    try:
        if args.warmup:
            run_trial(
                runtime,
                request,
                args.output / "warmup",
                args.workspace,
                reference,
                warmup=True,
            )
        for index in range(args.trials):
            metrics = run_trial(
                runtime,
                request,
                args.output / f"trial-{index + 1:03}",
                args.workspace,
                reference,
            )
            print(json.dumps(asdict(metrics)))
    finally:
        runtime.close()
