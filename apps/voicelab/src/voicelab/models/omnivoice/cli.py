"""Speech synthesis with retained audio and measurements."""

import argparse
from pathlib import Path

from voicelab.launch.devices import parse_device
from voicelab.runs.synthesis.cli import RunArguments, add_run_arguments, run
from voicelab.runs.synthesis.execution import ModelEvidence
from voicelab_models.audio import ReferenceAudio
from voicelab_models.omnivoice.model import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RATE,
    Request,
)


class Arguments(RunArguments):
    language: str
    instruction: str | None
    steps: int
    speed: float
    seed: int
    duration: float | None
    normalize: bool
    denoise: bool
    preprocess_prompt: bool
    postprocess_output: bool
    guidance_scale: float
    t_shift: float
    layer_penalty_factor: float
    position_temperature: float
    class_temperature: float
    audio_chunk_duration: float
    audio_chunk_threshold: float
    pad_duration: float
    fade_duration: float
    reference_audio: Path | None
    reference_text: str | None
    device: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="omnivoice synthesis and RTF measurements"
    )
    add_run_arguments(parser)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument(
        "--normalize", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--denoise", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--preprocess-prompt", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--postprocess-output", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--t-shift", type=float, default=0.1)
    parser.add_argument("--layer-penalty-factor", type=float, default=5.0)
    parser.add_argument("--position-temperature", type=float, default=5.0)
    parser.add_argument("--class-temperature", type=float, default=0.0)
    parser.add_argument("--audio-chunk-duration", type=float, default=15.0)
    parser.add_argument("--audio-chunk-threshold", type=float, default=30.0)
    parser.add_argument("--pad-duration", type=float, default=0.1)
    parser.add_argument("--fade-duration", type=float, default=0.1)
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--reference-text")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args(namespace=Arguments())
    request = Request.model_validate(
        {
            "text": args.text,
            "language": None if args.language == "auto" else args.language,
            "instruction": args.instruction,
            "steps": args.steps,
            "speed": args.speed,
            "seed": args.seed,
            "duration": args.duration,
            "normalize": args.normalize,
            "denoise": args.denoise,
            "preprocess_prompt": args.preprocess_prompt,
            "postprocess_output": args.postprocess_output,
            "guidance_scale": args.guidance_scale,
            "t_shift": args.t_shift,
            "layer_penalty_factor": args.layer_penalty_factor,
            "position_temperature": args.position_temperature,
            "class_temperature": args.class_temperature,
            "audio_chunk_duration": args.audio_chunk_duration,
            "audio_chunk_threshold": args.audio_chunk_threshold,
            "pad_duration": args.pad_duration,
            "fade_duration": args.fade_duration,
            "ref_audio": ReferenceAudio.from_path(args.reference_audio)
            if args.reference_audio is not None
            else None,
            "ref_text": args.reference_text,
        }
    )
    from voicelab_inference.omnivoice.torch.runtime import OmniVoiceTorch

    OmniVoiceTorch.validate_dependencies(request)
    runtime = OmniVoiceTorch(parse_device(args.device))
    evidence = ModelEvidence(
        MODEL_ID,
        MODEL_REVISION,
        CHECKPOINT_ID,
        CHECKPOINT_REVISION,
        SAMPLE_RATE,
        ("torch", "transformers", "numpy", "omnivoice"),
    )
    run(parser, args, request, runtime, evidence)
