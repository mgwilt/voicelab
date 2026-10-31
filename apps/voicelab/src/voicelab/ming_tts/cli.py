"""Run ming_tts experiments through the shared artifact and measurement lifecycle."""

import argparse
from pathlib import Path

from voicelab.synthesis.cli import RunArguments, add_run_arguments, run
from voicelab.synthesis.experiments import AuxiliaryCheckpoint, ModelEvidence
from voicelab_models.audio import ReferenceAudio
from voicelab_models.ming_tts.model import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RATE,
    Request,
)


class Arguments(RunArguments):
    instruction: str | None
    language: str
    voice: str | None
    style: str | None
    emotion: str | None
    dialect: str | None
    speed: str | None
    pitch: str | None
    volume: str | None
    flow_steps: int
    temperature: float
    sigma: float
    ref_audio: Path | None
    ref_text: str | None
    trim_reference_silence: bool
    seed: int
    max_frames: int
    cfg_scale: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ming_tts synthesis and RTF measurements"
    )
    add_run_arguments(parser)
    parser.add_argument("--instruction", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--ref-text")
    parser.add_argument(
        "--trim-reference-silence", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--language", default="en", choices=("zh", "en", "ja", "ko", "de", "fr", "es")
    )
    for control in ("voice", "style", "emotion", "dialect", "speed", "pitch", "volume"):
        parser.add_argument(f"--{control}")
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=0.25)
    args = parser.parse_args(namespace=Arguments())
    request = Request.model_validate(
        {
            "text": args.text,
            "ref_audio": ReferenceAudio.from_path(args.ref_audio)
            if args.ref_audio
            else None,
            "ref_text": args.ref_text,
            "trim_reference_silence": args.trim_reference_silence,
            "instruction": args.instruction,
            "language": args.language,
            "voice": args.voice,
            "style": args.style,
            "emotion": args.emotion,
            "dialect": args.dialect,
            "speed": args.speed,
            "pitch": args.pitch,
            "volume": args.volume,
            "flow_steps": args.flow_steps,
            "temperature": args.temperature,
            "sigma": args.sigma,
            "seed": args.seed,
            "max_frames": args.max_frames,
            "cfg_scale": args.cfg_scale,
        }
    )
    from voicelab_inference.ming_tts.mlx.runtime import MingTTSMLX

    runtime = MingTTSMLX()
    evidence = ModelEvidence(
        MODEL_ID,
        MODEL_REVISION,
        CHECKPOINT_ID,
        CHECKPOINT_REVISION,
        SAMPLE_RATE,
        ("mlx", "mlx-audio", "numpy", "onnxruntime", "torchaudio"),
        auxiliary_checkpoints=(
            AuxiliaryCheckpoint(
                "speaker_encoder", MODEL_ID, MODEL_REVISION, "campplus.onnx"
            ),
        ),
    )
    run(parser, args, request, runtime, evidence)
