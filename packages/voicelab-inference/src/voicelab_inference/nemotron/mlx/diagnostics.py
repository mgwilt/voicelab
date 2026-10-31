"""Runtime feasibility checks without loading model weights."""

import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version


@dataclass(frozen=True)
class RuntimeStatus:
    apple_silicon: bool
    metal_available: bool
    mlx_vlm_version: str | None
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.apple_silicon and self.metal_available and self.error is None


def inspect_runtime() -> RuntimeStatus:
    supported = platform.system() == "Darwin" and platform.machine() == "arm64"
    if not supported:
        return RuntimeStatus(False, False, None, "Requires an Apple Silicon Mac.")
    try:
        runtime_version = version("mlx-vlm")
        import mlx.core as mx
        from mlx_vlm.models.nemotron_voicechat import Model

        if Model.__name__ != "Model":
            raise RuntimeError("Unexpected VoiceChat implementation.")
        return RuntimeStatus(True, mx.metal.is_available(), runtime_version)
    except (ImportError, PackageNotFoundError) as exc:
        return RuntimeStatus(True, False, None, str(exc))
