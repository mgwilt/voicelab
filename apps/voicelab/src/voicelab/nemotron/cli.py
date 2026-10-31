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
    tools_json: Path | None
    api_response_json: Path | None


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
    run.add_argument(
        "--tools-json", type=Path, help="OpenAI-format tool definitions JSON"
    )
    run.add_argument(
        "--api-response-json", type=Path, help="Supplied tool_name/response JSON"
    )
    args = parser.parse_args(namespace=Arguments())
    if args.command == "doctor":
        return doctor()
    settings = GenerationSettings(args.prompt, args.seed, args.trailing_silence)
    if (args.tools_json is None) != (args.api_response_json is None):
        parser.error("--tools-json and --api-response-json must be supplied together")
    if args.tools_json is not None and args.api_response_json is not None:
        from voicelab.nemotron.tool_experiments import (
            read_tool_settings,
            run_tool_experiment,
        )

        tools = read_tool_settings(args.tools_json, args.api_response_json)
        print(
            run_tool_experiment(
                args.input, args.output, settings, tools, args.workspace
            )
        )
        return 0
    from voicelab_inference.nemotron.mlx.runtime import NemotronMLX

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
