"""Validate explicit device choices without importing an inference framework."""

from typing import Literal


def parse_device(value: str) -> Literal["cpu", "mps", "cuda"]:
    match value:
        case "cpu":
            return "cpu"
        case "mps":
            return "mps"
        case "cuda":
            return "cuda"
        case _:
            raise ValueError("Expected cpu, mps, or cuda")
