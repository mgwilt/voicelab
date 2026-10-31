"""Synchronized frame-processing metrics; transport delay is separate."""

from dataclasses import dataclass, field

from voicelab_models.nemotron.streaming import FRAME_SECONDS


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass
class FrameMetrics:
    milliseconds: list[float] = field(default_factory=list)

    def add(self, seconds: float) -> None:
        self.milliseconds.append(seconds * 1000)

    def summary(self) -> dict[str, float | int | None]:
        frames = len(self.milliseconds)
        if not frames:
            return {"frames": 0, "rtf": None, "p50_ms": None, "p95_ms": None}
        return {
            "frames": frames,
            "input_seconds": frames * FRAME_SECONDS,
            "processing_seconds": sum(self.milliseconds) / 1000,
            "rtf": sum(self.milliseconds) / (frames * FRAME_SECONDS * 1000),
            "p50_ms": percentile(self.milliseconds, 0.5),
            "p95_ms": percentile(self.milliseconds, 0.95),
            "max_ms": max(self.milliseconds),
            "over_budget_frames": sum(
                ms > FRAME_SECONDS * 1000 for ms in self.milliseconds
            ),
        }
