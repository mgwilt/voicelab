"""Convenience entry point for the Nemotron offline MLX experiment."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from voicelab.nemotron.experiments import run_experiment
from voicelab_core.contracts import GenerationSettings
from voicelab_models.nemotron.model import MLX_8BIT


def doctor() -> int:
    """Probe runtime availability without loading or downloading a checkpoint."""
    from voicelab_inference.nemotron.mlx.diagnostics import inspect_runtime

    status = inspect_runtime()
    print(
        json.dumps(
            {"runtime": asdict(status), "checkpoint": asdict(MLX_8BIT)}, indent=2
        )
    )
    return 0 if status.ready else 1


class Arguments(argparse.Namespace):
    command: str
    input: Path
    output: Path
    prompt: str
    seed: int
    trailing_silence: float
    workspace: Path


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="voicelab-nemotron",
        description="Nemotron offline MLX experiments on Apple Silicon.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor", help="Check Apple Silicon and MLX without loading weights"
    )
    run = commands.add_parser("run", help="Run one offline Nemotron MLX experiment")
    run.add_argument("--input", type=Path, required=True, help="Mono PCM16 WAV, 16 kHz")
    run.add_argument(
        "--output", type=Path, required=True, help="New directory for run artifacts"
    )
    run.add_argument("--prompt", default=GenerationSettings().prompt)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--trailing-silence", type=float, default=3.0)
    run.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args(namespace=Arguments())
    if args.command == "doctor":
        return doctor()
    from voicelab_inference.nemotron.mlx.runtime import NemotronMLX

    settings = GenerationSettings(args.prompt, args.seed, args.trailing_silence)
    manifest = run_experiment(
        args.input,
        args.output,
        MLX_8BIT,
        settings,
        lambda: NemotronMLX(MLX_8BIT),
        backend_name="nemotron/mlx-vlm-reference",
        workspace=args.workspace,
    )
    print(manifest)
    return 0
