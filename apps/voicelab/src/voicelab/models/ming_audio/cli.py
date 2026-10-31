"""Ming Audio task selection, conversation context, and reference voice inputs."""

import argparse
import base64
from pathlib import Path

from pydantic import TypeAdapter

from voicelab_models.audio import ReferenceAudio
from voicelab_models.ming_audio.model import HistoryTurn, Request, Task


class Arguments(argparse.Namespace):
    input: Path | None
    output: Path
    task: Task
    prompt: str
    voice: str
    max_tokens: int
    seed: int
    max_decode_steps: int
    cfg_scale: float
    sigma: float
    temperature: float
    flow_steps: int
    reference_audio: Path | None
    reference_text: str | None
    history: Path | None
    system_prompt: str | None
    thinking: bool
    keep_reference_silence: bool


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Ming Audio Tiny locally with MLX."
    )
    parser.add_argument(
        "--task",
        choices=("spoken_dialogue", "audio_understanding", "asr", "text", "tts"),
        default="spoken_dialogue",
    )
    parser.add_argument("--input", type=Path, help="Mono PCM16 WAV16k, <=30s")
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument(
        "--prompt", "--text", default="Please answer the question in the audio."
    )
    parser.add_argument("--voice", default="US_Stella")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-decode-steps", type=int, default=375)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument(
        "--reference-audio",
        type=Path,
        help="Reference voice WAV; replaces preset voice",
    )
    parser.add_argument(
        "--reference-text", help="Optional transcript of the voice reference"
    )
    parser.add_argument("--system-prompt")
    parser.add_argument(
        "--history", type=Path, help="JSON array of prior user/assistant text turns"
    )
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument(
        "--keep-reference-silence",
        action="store_true",
        help="Preserve the acoustic reference tail for continuation runs",
    )
    args = parser.parse_args(namespace=Arguments())
    audio: str | None = None
    if args.input is not None:
        if args.input.stat().st_size > 1025536:
            parser.error("Input WAV exceeds the 30-second audio limit.")
        audio = base64.b64encode(args.input.read_bytes()).decode("ascii")
    history: tuple[HistoryTurn, ...] = ()
    if args.history is not None:
        if args.history.stat().st_size > 100000:
            parser.error("History exceeds 100 KiB.")
        history = TypeAdapter(tuple[HistoryTurn, ...]).validate_json(
            args.history.read_bytes()
        )
    request = Request(
        task=args.task,
        audio_wav_base64=audio,
        prompt=args.prompt,
        voice=args.voice,
        max_tokens=args.max_tokens,
        seed=args.seed,
        max_decode_steps=args.max_decode_steps,
        cfg_scale=args.cfg_scale,
        sigma=args.sigma,
        temperature=args.temperature,
        flow_steps=args.flow_steps,
        system_prompt=args.system_prompt,
        history=history,
        thinking=args.thinking,
        reference_audio=ReferenceAudio.from_path(args.reference_audio)
        if args.reference_audio is not None
        else None,
        reference_text=args.reference_text,
        trim_reference_silence=not args.keep_reference_silence,
    )
    from voicelab.models.ming_audio.runs import execute_run

    print(execute_run(request, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
