"""Portable runtime evidence; callers save these values only in local reports."""

import hashlib
import platform
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


@dataclass(frozen=True)
class RuntimeEvidence:
    python: str
    operating_system: str
    architecture: str
    processor: str
    versions: dict[str, str]
    source_sha256: dict[str, str]


def runtime_evidence(
    workspace: Path, distributions: tuple[str, ...]
) -> RuntimeEvidence:
    processor = platform.processor() or "unavailable"
    if platform.system() == "Darwin":
        probe = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            processor = probe.stdout.strip()
    versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = "not-installed"
    paths = [
        workspace / "uv.lock",
        workspace / "pyproject.toml",
        workspace / "voicelab",
    ]
    for root in (workspace / "apps", workspace / "packages", workspace / "third_party"):
        paths.extend(
            path
            for path in root.rglob("*")
            if (path.suffix in {".py", ".toml", ".js"} or path.name == "uv.lock")
            and not {"node_modules", ".venv", "__pycache__"}.intersection(path.parts)
        )
    hashes = {
        str(path.relative_to(workspace)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
        if path.is_file()
    }
    return RuntimeEvidence(
        platform.python_version(),
        platform.platform(),
        platform.machine(),
        processor,
        versions,
        hashes,
    )
