"""Execute offline inference and retain reproducible evidence."""

import hashlib
import json
import platform
import subprocess
import time
import wave
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from voicelab_core.audio import inspect_input
from voicelab_core.contracts import GenerationSettings, OfflineSpeechBackend
from voicelab_models.nemotron.model import SOURCE_MODEL, Checkpoint


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def execute_run(
    source: Path,
    output: Path,
    checkpoint: Checkpoint,
    settings: GenerationSettings,
    factory: Callable[[], OfflineSpeechBackend],
    *,
    backend_name: str,
    workspace: Path,
) -> Path:
    """Measure a single load and generation; this is not a streaming latency test."""
    audio_info = inspect_input(source)
    output.mkdir(parents=True, exist_ok=False)
    packages: dict[str, str] = {}
    for name in ("voicelab", "mlx-vlm", "mlx", "mlx-metal", "mlx-audio", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    hardware = platform.machine()
    if platform.system() == "Darwin":
        probe = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            capture_output=True,
            check=False,
        )
        hardware = probe.stdout.strip() if probe.returncode == 0 else "unavailable"
    sources = sorted(
        path
        for root in (workspace / "packages", workspace / "apps")
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".toml"}
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "model": SOURCE_MODEL,
        "checkpoint": asdict(checkpoint),
        "backend": backend_name,
        "mode": "offline-single-turn",
        "settings": asdict(settings),
        "input": {
            "path": str(source.resolve()),
            "sha256": fingerprint(source),
            "duration_seconds": audio_info.duration_seconds,
        },
        "host": {
            "os": platform.platform(),
            "chip": hardware,
            "architecture": platform.machine(),
            "python": platform.python_version(),
        },
        "packages": packages,
        "uv_lock_sha256": fingerprint(workspace / "uv.lock"),
        "source_sha256": {
            str(p.relative_to(workspace)): fingerprint(p) for p in sources
        },
        "measurement_notes": [
            "One trial; no warmup. Load may include download or cache effects.",
            "Generation includes preprocessing, synchronization and WAV serialization.",
            "RTF uses input duration including configured trailing silence.",
            "No time-to-first-audio, streaming latency, quality or memory measurement.",
        ],
    }
    manifest = output / "run.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    try:
        start = time.perf_counter()
        backend = factory()
        backend.synchronize()
        report["load_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        result = backend.generate(source, output / "response.wav", settings)
        backend.synchronize()
        elapsed = time.perf_counter() - start
        with wave.open(str(result.audio_path), "rb") as response:
            output_seconds = response.getnframes() / response.getframerate()
        report.update(
            {
                "status": "completed",
                "generation_seconds": elapsed,
                "input_timeline_rtf": elapsed
                / (audio_info.duration_seconds + settings.trailing_silence_seconds),
                "output_duration_seconds": output_seconds,
                "output_audio_sha256": fingerprint(result.audio_path),
                "output_sample_rate": result.sample_rate,
                "text": result.text,
                "user_transcript": result.user_transcript,
                "function_token_ids": result.function_token_ids,
            }
        )
    except BaseException as exc:
        report.update(
            {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        )
        raise
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        manifest.write_text(json.dumps(report, indent=2) + "\n")
    return manifest
