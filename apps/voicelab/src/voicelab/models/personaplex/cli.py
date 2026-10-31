"""PersonaPlex local duplex capture arguments."""

import argparse
from pathlib import Path

from voicelab_models.audio import ReferenceAudio
from voicelab_models.personaplex.model import Request


class Arguments(argparse.Namespace):
    input: Path
    output: Path
    prompt: str
    voice: str
    trailing_silence: float
    reference_audio: Path | None
    seed: int
    text_temperature: float
    audio_temperature: float
    text_top_k: int
    audio_top_k: int


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture a PersonaPlex duplex session."
    )
    parser.add_argument("--input", type=Path, required=True, help="Mono PCM16 WAV24k")
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--prompt", default=Request().prompt)
    parser.add_argument("--voice", default="NATF2.pt")
    parser.add_argument("--trailing-silence", type=float, default=3.0)
    parser.add_argument(
        "--reference-audio",
        type=Path,
        help="Reference voice WAV; replaces preset voice",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-temperature", type=float, default=0.7)
    parser.add_argument("--audio-temperature", type=float, default=0.8)
    parser.add_argument("--text-top-k", type=int, default=25)
    parser.add_argument("--audio-top-k", type=int, default=250)
    args = parser.parse_args(namespace=Arguments())
    request = Request(
        prompt=args.prompt,
        voice=args.voice,
        seed=args.seed,
        text_temperature=args.text_temperature,
        audio_temperature=args.audio_temperature,
        text_top_k=args.text_top_k,
        audio_top_k=args.audio_top_k,
        reference_audio=ReferenceAudio.from_path(args.reference_audio)
        if args.reference_audio is not None
        else None,
    )
    from voicelab.models.personaplex.runs import execute_run

    print(execute_run(args.input, args.output, request, args.trailing_silence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
