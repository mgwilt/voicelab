"""Speech synthesis with retained audio and measurements."""

import argparse
from pathlib import Path

from voicelab.runs.synthesis.cli import RunArguments, add_run_arguments, run
from voicelab.runs.synthesis.execution import ModelEvidence
from voicelab_models.audio import ReferenceAudio
from voicelab_models.audio8.model import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RATE,
    Request,
)


class Arguments(RunArguments):
    ref_audio: Path | None
    ref_text: str | None
    seed: int
    temperature: float
    max_frames: int
    top_p: float
    top_k: int
    do_sample: bool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="audio8 synthesis and RTF measurements"
    )
    add_run_arguments(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--ref-text")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--do-sample", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(namespace=Arguments())
    request = Request.model_validate(
        {
            "text": args.text,
            "ref_audio": ReferenceAudio.from_path(args.ref_audio)
            if args.ref_audio
            else None,
            "ref_text": args.ref_text,
            "seed": args.seed,
            "temperature": args.temperature,
            "max_frames": args.max_frames,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "do_sample": args.do_sample,
        }
    )
    from voicelab_inference.audio8.mlx.runtime import Audio8MLX

    runtime = Audio8MLX()
    evidence = ModelEvidence(
        MODEL_ID,
        MODEL_REVISION,
        CHECKPOINT_ID,
        CHECKPOINT_REVISION,
        SAMPLE_RATE,
        ("mlx", "mlx-audio", "numpy"),
    )
    run(parser, args, request, runtime, evidence)
