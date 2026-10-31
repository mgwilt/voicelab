"""Speech synthesis with retained audio and measurements."""

import argparse
from pathlib import Path

from voicelab.runs.synthesis.cli import RunArguments, add_run_arguments, run
from voicelab.runs.synthesis.execution import AuxiliaryCheckpoint, ModelEvidence
from voicelab_models.audio import ReferenceAudio
from voicelab_models.maya.model import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    CODEC_FILE,
    CODEC_ID,
    CODEC_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RATE,
    SPEAKER_ID,
    SPEAKER_REVISION,
    Request,
)


class Arguments(RunArguments):
    language: str
    voice: str | None
    conditioning: str
    ref_language: str | None
    max_frames: int
    min_frames: int
    seed: int
    normalize: bool
    temperature: float
    top_k: int
    top_p: float
    depth_temperature: float
    depth_top_k: int
    presence_penalty: float
    presence_window: int
    streaming_interval: float
    reference_audio: Path | None
    reference_text: str | None


def main() -> None:
    parser = argparse.ArgumentParser(description="maya synthesis and RTF measurements")
    add_run_arguments(parser)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--voice", type=str, default=None)
    parser.add_argument("--conditioning", type=str, default="icl_spk")
    parser.add_argument("--ref-language", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=750)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--normalize", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--depth-temperature", type=float, default=0.7)
    parser.add_argument("--depth-top-k", type=int, default=64)
    parser.add_argument("--presence-penalty", type=float, default=0.8)
    parser.add_argument("--presence-window", type=int, default=12)
    parser.add_argument("--streaming-interval", type=float, default=0.32)
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--reference-text")
    args = parser.parse_args(namespace=Arguments())
    request = Request.model_validate(
        {
            "text": args.text,
            "language": args.language,
            "voice": args.voice,
            "conditioning": args.conditioning,
            "ref_language": args.ref_language,
            "max_frames": args.max_frames,
            "min_frames": args.min_frames,
            "seed": args.seed,
            "normalize": args.normalize,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "depth_temperature": args.depth_temperature,
            "depth_top_k": args.depth_top_k,
            "presence_penalty": args.presence_penalty,
            "presence_window": args.presence_window,
            "streaming_interval": args.streaming_interval,
            "ref_audio": ReferenceAudio.from_path(args.reference_audio)
            if args.reference_audio is not None
            else None,
            "ref_text": args.reference_text,
        }
    )
    from voicelab_inference.maya.mlx.runtime import MayaMLX

    runtime = MayaMLX()
    evidence = ModelEvidence(
        MODEL_ID,
        MODEL_REVISION,
        CHECKPOINT_ID,
        CHECKPOINT_REVISION,
        SAMPLE_RATE,
        ("numpy", "mlx", "mlx-audio", "transformers", "tokenizers"),
        auxiliary_checkpoints=(
            AuxiliaryCheckpoint("codec", CODEC_ID, CODEC_REVISION, CODEC_FILE),
            AuxiliaryCheckpoint(
                "speaker", SPEAKER_ID, SPEAKER_REVISION, "model.safetensors"
            ),
        ),
    )
    run(parser, args, request, runtime, evidence)
