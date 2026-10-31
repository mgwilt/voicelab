"""Common run options and resource ownership; model parsers retain their inputs."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel

from voicelab.runs.synthesis.execution import ModelEvidence, SynthesisBackend, run_trial


class RunArguments(argparse.Namespace):
    text: str
    output: Path
    workspace: Path
    trials: int
    warmup: bool


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--text", required=True, help="Text to speak")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--warmup", action="store_true")


def run[Request: BaseModel](
    parser: argparse.ArgumentParser,
    args: RunArguments,
    request: Request,
    runtime: SynthesisBackend[Request],
    model: ModelEvidence,
) -> None:
    if not 1 <= args.trials <= 100:
        parser.error("trials must be between 1 and 100")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        try:
            runtime.load()
        except BaseException as exc:
            (args.output / "load.json").write_text(
                json.dumps({"status": "failed", "error": str(exc)}) + "\n"
            )
            raise
        (args.output / "load.json").write_text(
            json.dumps(
                {"status": "loaded", "load_seconds": time.perf_counter() - started}
            )
            + "\n"
        )
        if args.warmup:
            run_trial(
                runtime,
                request,
                model,
                args.output / "warmup",
                args.workspace,
                warmup=True,
            )
        for index in range(args.trials):
            metrics = run_trial(
                runtime,
                request,
                model,
                args.output / f"trial-{index + 1:03}",
                args.workspace,
            )
            print(json.dumps(asdict(metrics)))
    finally:
        runtime.close()
