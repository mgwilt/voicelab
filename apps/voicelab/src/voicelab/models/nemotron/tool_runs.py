"""Evidence for one local, user-supplied tool-response run."""

import hashlib
import json
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from voicelab.models.nemotron.runs import fingerprint
from voicelab_core.audio import inspect_input
from voicelab_core.contracts import GenerationSettings
from voicelab_models.nemotron.model import MLX_8BIT
from voicelab_models.nemotron.tools import (
    SOURCE_REVISION,
    SuppliedResponse,
    Tool,
    ToolSettings,
)


def read_tool_settings(tools_file: Path, response_file: Path) -> ToolSettings:
    for path in (tools_file, response_file):
        if path.stat().st_size > 16000:
            raise ValueError("Tool JSON files must not exceed 16000 bytes.")
    return ToolSettings(
        tools=TypeAdapter(tuple[Tool, ...]).validate_json(tools_file.read_bytes()),
        api_response=SuppliedResponse.model_validate_json(response_file.read_bytes()),
    )


def execute_tool_run(
    source: Path,
    output: Path,
    settings: GenerationSettings,
    tools: ToolSettings,
    workspace: Path,
) -> Path:
    audio = inspect_input(source)
    if not 0 <= settings.seed <= 2147483647 or len(settings.prompt) > 2000:
        raise ValueError(
            "Seed must be 0..2147483647 and prompt at most 2000 characters."
        )
    if audio.duration_seconds > 30 or settings.trailing_silence_seconds > 30:
        raise ValueError("Input and trailing silence must each be at most 30 seconds.")
    if audio.duration_seconds + settings.trailing_silence_seconds > 60:
        raise ValueError("Tool runs support at most 60 seconds including silence.")
    if not settings.prompt.isascii():
        raise ValueError("Nemotron system prompts must be ASCII.")
    output.mkdir(parents=True, exist_ok=False)
    roots = (
        "packages/voicelab-inference/src/voicelab_inference/nemotron",
        "packages/voicelab-models/src/voicelab_models/nemotron",
        "apps/voicelab/src/voicelab/models/nemotron",
    )
    sources = sorted(
        path for root in roots for path in (workspace / root).rglob("*.py")
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "backend": "nemotron/mlx-local-tool-response",
        "mode": "offline-two-pass-supplied-response",
        "completion_basis": "EOTR plus continuation text; not utterance EOS",
        "source_revision": SOURCE_REVISION,
        "checkpoint": asdict(MLX_8BIT),
        "mlx_vlm_version": version("mlx-vlm"),
        "mlx_version": version("mlx"),
        "uv_lock_sha256": fingerprint(workspace / "uv.lock"),
        "source_sha256": {
            str(path.relative_to(workspace)): fingerprint(path) for path in sources
        },
        "settings": asdict(settings),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_seconds": audio.duration_seconds,
        "tool_settings_sha256": hashlib.sha256(
            tools.model_dump_json().encode()
        ).hexdigest(),
        "supplied_tool_name": tools.api_response.tool_name,
        "measurement_notes": [
            "Both offline passes count toward RTF; checkpoint loading is separate.",
            "RTF denominator is original input duration plus trailing silence.",
            "WAV retains the expanded timeline, including inserted silence.",
            "No tool is executed. The result is supplied from a local JSON file.",
        ],
    }
    manifest = output / "run.json"
    try:
        from voicelab_inference.nemotron.mlx.runtime import NemotronMLX

        started = time.perf_counter()
        runtime = NemotronMLX(MLX_8BIT)
        report["load_seconds"] = time.perf_counter() - started
        result = runtime.generate_with_tool(
            source, output / "response.wav", settings, tools
        )
        values = asdict(result)
        values["call"] = {
            "name": result.call.call.name,
            "arguments": result.call.call.arguments,
            "start_frame": result.call.start_frame,
            "end_frame": result.call.end_frame,
            "token_count": result.call.token_count,
        }
        report.update(values)
        elapsed = result.detection_seconds + result.continuation_seconds
        report["generation_seconds"] = elapsed
        report["input_timeline_rtf"] = elapsed / (
            audio.duration_seconds + settings.trailing_silence_seconds
        )
        report["status"] = "completed" if result.completed else "incomplete"
        if report["status"] != "completed":
            raise RuntimeError(
                "Tool response did not produce EOTR and a continued answer."
            )
    except BaseException as exc:
        if report["status"] != "incomplete":
            report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        if isinstance(exc, ValidationError):
            report["error"] = "; ".join(
                error["msg"]
                for error in exc.errors(
                    include_input=False, include_context=False, include_url=False
                )
            )
        else:
            report["error"] = str(exc)[:1000]
        raise
    finally:
        manifest.write_text(json.dumps(report, indent=2) + "\n")
    return manifest
