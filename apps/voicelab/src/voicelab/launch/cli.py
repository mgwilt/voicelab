"""One user-facing command, with explicit model-specific experiment adapters."""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from voicelab.launch.workspace import Workspace


@dataclass(frozen=True)
class RuntimeCommand:
    name: str
    executable: str
    prefix: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelCommand:
    name: str
    label: str
    runtimes: tuple[RuntimeCommand, ...]


class Arguments(argparse.Namespace):
    command: str | None
    config: Path | None
    model_command: ModelCommand
    runtime: str


def commands() -> tuple[ModelCommand, ...]:
    """Application composition, not runtime discovery or an inference registry."""
    return (
        ModelCommand(
            "nemotron",
            "Voice conversation",
            (RuntimeCommand("mlx", "voicelab-nemotron", ("run",)),),
        ),
        ModelCommand(
            "sesame",
            "Contextual speech synthesis",
            (RuntimeCommand("torch", "voicelab-sesame"),),
        ),
        ModelCommand(
            "breeze",
            "Instruction-guided speech synthesis",
            (RuntimeCommand("mlx", "voicelab-breeze"),),
        ),
    )


def parser_for(models: tuple[ModelCommand, ...]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voicelab",
        description="Voice model experiments. Run ./voicelab to start the website.",
        epilog=(
            "Examples: ./voicelab | ./voicelab models | "
            "./voicelab run breeze --text 'Hello'"
        ),
    )
    parser.set_defaults(config=None)
    subcommands = parser.add_subparsers(dest="command")
    serve = subcommands.add_parser("serve", help="Start the shared website (default)")
    serve.add_argument("--config", type=Path, help="Local server configuration")
    subcommands.add_parser("models", help="List models and their implemented runtimes")
    run = subcommands.add_parser(
        "run", help="Run a model experiment; options vary by model"
    )
    choices = run.add_subparsers(dest="model", required=True)
    for model in models:
        item = choices.add_parser(model.name, help=model.label, add_help=False)
        item.add_argument(
            "--runtime",
            choices=tuple(runtime.name for runtime in model.runtimes),
            default=model.runtimes[0].name,
        )
        item.set_defaults(model_command=model)
    return parser


def experiment_arguments(
    model: ModelCommand, runtime: RuntimeCommand, arguments: list[str]
) -> list[str]:
    result = [*runtime.prefix, *arguments]
    if (
        "--help" not in arguments
        and "-h" not in arguments
        and not any(
            value == "--output" or value.startswith("--output=") for value in arguments
        )
    ):
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        result.extend(["--output", f"artifacts/runs/{model.name}-{stamp}"])
    return result


def main(argv: list[str] | None = None) -> int:
    models = commands()
    parser = parser_for(models)
    args, forwarded = parser.parse_known_args(argv, namespace=Arguments())
    command = args.command or "serve"
    if forwarded and command != "run":
        parser.error("unrecognized arguments: " + " ".join(forwarded))
    if command == "models":
        for model in models:
            runtimes = ", ".join(runtime.name for runtime in model.runtimes)
            print(f"{model.name:10} {runtimes:8} {model.label}")
        return 0
    root = Path(__file__).resolve().parents[5]
    workspace = Workspace(root)
    try:
        workspace.prepare_python()
        if command == "serve":
            workspace.prepare_frontend()
            config = args.config
            options = (
                ["--config", str(config.resolve())] if isinstance(config, Path) else []
            )
            workspace.execute("voicelab-web", options)
        else:
            model = args.model_command
            runtime = next(
                runtime for runtime in model.runtimes if runtime.name == args.runtime
            )
            workspace.execute(
                runtime.executable, experiment_arguments(model, runtime, forwarded)
            )
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Voice Lab: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
