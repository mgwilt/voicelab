"""Model identity and constraints, independent of inference frameworks."""

from dataclasses import dataclass

SOURCE_MODEL = "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"
MODEL_LICENSE = "OpenMDW-1.1"
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 22050


@dataclass(frozen=True)
class Checkpoint:
    repository: str
    revision: str
    precision: str

    def __post_init__(self) -> None:
        if len(self.revision) != 40 or any(
            c not in "0123456789abcdef" for c in self.revision
        ):
            raise ValueError("Revision must be an immutable 40-character commit SHA.")


MLX_8BIT = Checkpoint(
    repository="mlx-community/NemotronLabs-VoiceChat-11B-8bit",
    revision="446fc83f6cde720e7fd9ad7e29d7acb3adbb8967",
    precision="8bit",
)
