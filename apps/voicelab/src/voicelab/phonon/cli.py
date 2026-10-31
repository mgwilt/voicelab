"""Run phonon experiments through the shared artifact and measurement lifecycle."""

import argparse

from voicelab.synthesis.cli import RunArguments, add_run_arguments, run
from voicelab.synthesis.devices import parse_device
from voicelab.synthesis.experiments import AuxiliaryCheckpoint, ModelEvidence
from voicelab_models.phonon.model import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    CODEC_ID,
    CODEC_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RATE,
    Request,
)
from voicelab_models.phonon.voices import VOICE_REPOSITORY, VOICE_REVISION


class Arguments(RunArguments):
    seed: int
    max_frames: int
    device: str
    voice: str
    temperature: float
    streaming: bool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="phonon synthesis and RTF measurements"
    )
    add_run_arguments(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=750)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--voice", default="default")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument(
        "--streaming", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(namespace=Arguments())
    request = Request.model_validate(
        {
            "text": args.text,
            "seed": args.seed,
            "max_frames": args.max_frames,
            "voice": args.voice,
            "temperature": args.temperature,
            "streaming": args.streaming,
        }
    )
    from voicelab_inference.phonon.torch.runtime import PhononTorch

    runtime = PhononTorch(parse_device(args.device))
    evidence = ModelEvidence(
        MODEL_ID,
        MODEL_REVISION,
        CHECKPOINT_ID,
        CHECKPOINT_REVISION,
        SAMPLE_RATE,
        ("torch", "transformers", "numpy", "tokenizers", "sentencepiece"),
        auxiliary_checkpoints=(
            AuxiliaryCheckpoint("codec", CODEC_ID, CODEC_REVISION),
            AuxiliaryCheckpoint("voice_catalog", VOICE_REPOSITORY, VOICE_REVISION),
        ),
    )
    run(parser, args, request, runtime, evidence)
