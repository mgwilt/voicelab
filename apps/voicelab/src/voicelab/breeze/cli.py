"""Breeze instruction synthesis and repeated RTF experiments."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from voicelab.breeze.experiments import run_trial
from voicelab_inference.breeze.mlx.runtime import BreezeMLX
from voicelab_models.breeze.synthesis import SynthesisRequest


class Arguments(argparse.Namespace):
    text: str
    instruction: str
    output: Path
    workspace: Path
    seed: int
    cfg_scale: float
    max_frames: int
    trials: int
    warmup: bool


def main() -> None:
    parser = argparse.ArgumentParser(description="Breeze MLX synthesis measurements")
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--instruction", default="A clear, natural conversational voice."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--max-frames", type=int, default=750)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args(namespace=Arguments())
    if not 1 <= args.trials <= 100:
        parser.error("trials must be between 1 and 100")
    request = SynthesisRequest(
        text=args.text,
        instruction=args.instruction,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        max_frames=args.max_frames,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = BreezeMLX()
    started = time.perf_counter()
    try:
        runtime.load()
    except Exception as exc:
        (args.output / "load.json").write_text(
            json.dumps({"status": "failed", "error": str(exc)})
        )
        raise
    (args.output / "load.json").write_text(
        json.dumps({"status": "loaded", "load_seconds": time.perf_counter() - started})
        + "\n"
    )
    if args.warmup:
        run_trial(runtime, request, args.output / "warmup", args.workspace, warmup=True)
    for index in range(args.trials):
        metrics = run_trial(
            runtime, request, args.output / f"trial-{index + 1:03}", args.workspace
        )
        print(json.dumps(asdict(metrics)))
