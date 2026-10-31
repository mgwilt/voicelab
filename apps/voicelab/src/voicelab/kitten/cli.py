"""Run kitten experiments through the shared artifact and measurement lifecycle."""

import argparse
from pathlib import Path

from voicelab.synthesis.cli import RunArguments, add_run_arguments, run
from voicelab.synthesis.devices import parse_device
from voicelab.synthesis.experiments import AuxiliaryCheckpoint, ModelEvidence
from voicelab_models.audio import ReferenceAudio
from voicelab_models.kitten.model import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    CODEC_FILE,
    CODEC_ID,
    CODEC_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RATE,
    Request,
)


class Arguments(RunArguments):
    voice: str | None
    preset: str
    max_tokens: int
    seed: int
    normalize: bool
    use_reference_prompt: bool | None
    temperature: float | None
    top_k: int | None
    top_p: float | None
    min_p: float | None
    repetition_penalty: float
    repetition_window: int | None
    token_run_penalty: float
    token_run_grace: int
    chunk_chars: int | None
    chunk_min_chars: int | None
    use_emotion: bool | None
    reference_audio: Path | None
    reference_text: str | None
    device: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="kitten synthesis and RTF measurements"
    )
    add_run_arguments(parser)
    parser.add_argument("--voice", type=str, default=None)
    parser.add_argument("--preset", type=str, default="stable")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--normalize", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--use-reference-prompt", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--repetition-window", type=int, default=None)
    parser.add_argument("--token-run-penalty", type=float, default=1.3)
    parser.add_argument("--token-run-grace", type=int, default=10)
    parser.add_argument("--chunk-chars", type=int, default=None)
    parser.add_argument("--chunk-min-chars", type=int, default=None)
    parser.add_argument(
        "--use-emotion", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--reference-text")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args(namespace=Arguments())
    request = Request.model_validate(
        {
            "text": args.text,
            "voice": args.voice,
            "preset": args.preset,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "normalize": args.normalize,
            "use_reference_prompt": args.use_reference_prompt,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "min_p": args.min_p,
            "repetition_penalty": args.repetition_penalty,
            "repetition_window": args.repetition_window,
            "token_run_penalty": args.token_run_penalty,
            "token_run_grace": args.token_run_grace,
            "chunk_chars": args.chunk_chars,
            "chunk_min_chars": args.chunk_min_chars,
            "use_emotion": args.use_emotion,
            "ref_audio": ReferenceAudio.from_path(args.reference_audio)
            if args.reference_audio is not None
            else None,
            "ref_text": args.reference_text,
        }
    )
    from voicelab_inference.kitten.torch.runtime import KittenTorch

    runtime = KittenTorch(parse_device(args.device))
    evidence = ModelEvidence(
        MODEL_ID,
        MODEL_REVISION,
        CHECKPOINT_ID,
        CHECKPOINT_REVISION,
        SAMPLE_RATE,
        ("torch", "transformers", "numpy", "kittenml", "tokenizers"),
        auxiliary_checkpoints=(
            AuxiliaryCheckpoint("codec", CODEC_ID, CODEC_REVISION, CODEC_FILE),
        ),
    )
    run(parser, args, request, runtime, evidence)
