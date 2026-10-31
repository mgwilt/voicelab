"""Run drama experiments through the shared artifact and measurement lifecycle."""

import argparse
from pathlib import Path

from voicelab.drama.history import load_previous_takes
from voicelab.synthesis.cli import RunArguments, add_run_arguments, run
from voicelab.synthesis.experiments import ModelEvidence
from voicelab_models.audio import ReferenceAudio
from voicelab_models.drama.model import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RATE,
    Request,
)


class Arguments(RunArguments):
    seed: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int | None
    streaming_interval: float
    first_chunk_frames: int | None
    reference_audio: list[Path]
    reference_text: list[str]
    previous_takes: Path | None


def main() -> None:
    parser = argparse.ArgumentParser(description="drama synthesis and RTF measurements")
    add_run_arguments(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--streaming-interval", type=float, default=0.32)
    parser.add_argument("--first-chunk-frames", type=int, default=None)
    parser.add_argument("--reference-audio", type=Path, action="append", default=[])
    parser.add_argument("--reference-text", action="append", default=[])
    parser.add_argument(
        "--previous-takes", type=Path, help="JSON list of earlier audio/text takes"
    )
    args = parser.parse_args(namespace=Arguments())
    if len(args.reference_audio) != len(args.reference_text):
        parser.error("Each --reference-audio requires one --reference-text")
    request = Request.model_validate(
        {
            "text": args.text,
            "previous_takes": load_previous_takes(args.previous_takes)
            if args.previous_takes is not None
            else [],
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "streaming_interval": args.streaming_interval,
            "first_chunk_frames": args.first_chunk_frames,
            "references": [
                {"audio": ReferenceAudio.from_path(path), "text": text}
                for path, text in zip(
                    args.reference_audio, args.reference_text, strict=True
                )
            ],
        }
    )
    from voicelab_inference.drama.mlx.runtime import DramaMLX

    runtime = DramaMLX()
    evidence = ModelEvidence(
        MODEL_ID,
        MODEL_REVISION,
        CHECKPOINT_ID,
        CHECKPOINT_REVISION,
        SAMPLE_RATE,
        ("numpy", "mlx", "mlx-audio"),
    )
    run(parser, args, request, runtime, evidence)
