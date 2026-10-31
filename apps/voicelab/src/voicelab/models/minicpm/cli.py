"""MiniCPM speech turn and duplex replay arguments."""

import argparse
from pathlib import Path

from pydantic import TypeAdapter

from voicelab_models.audio import ReferenceAudio
from voicelab_models.minicpm.model import HistoryTurn, Request


class Arguments(argparse.Namespace):
    history: Path | None
    mode: str
    input: Path | None
    text: str
    output: Path
    prompt: str
    reference_audio: Path | None
    seed: int
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    tts_temperature: float
    length_penalty: float
    listen_prob_scale: float
    max_speak_tokens: int
    trailing_silence: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local MiniCPM voice inference.")
    parser.add_argument("--mode", choices=("turn", "duplex"), default="turn")
    parser.add_argument(
        "--input", type=Path, help="Mono PCM16 WAV; duplex requires 16 kHz"
    )
    parser.add_argument("--text", default="")
    parser.add_argument(
        "--history", type=Path, help="JSON array of prior role/text pairs"
    )
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--prompt", default=Request().prompt)
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--tts-temperature", type=float, default=0.8)
    parser.add_argument("--length-penalty", type=float, default=1.1)
    parser.add_argument("--listen-prob-scale", type=float, default=1.0)
    parser.add_argument("--max-speak-tokens", type=int, default=20)
    parser.add_argument("--trailing-silence", type=float, default=5.0)
    args = parser.parse_args(namespace=Arguments())
    if args.mode == "duplex" and (args.input is None or args.text):
        parser.error("Duplex needs --input and does not accept --text.")
    request = Request(
        mode="full_duplex" if args.mode == "duplex" else "turn_based",
        text=args.text,
        history=TypeAdapter(tuple[HistoryTurn, ...]).validate_json(
            args.history.read_bytes(), strict=True
        )
        if args.history
        else (),
        audio=ReferenceAudio.from_path(args.input)
        if args.input is not None and args.mode == "turn"
        else None,
        prompt=args.prompt,
        reference_audio=ReferenceAudio.from_path(args.reference_audio)
        if args.reference_audio is not None
        else None,
        seed=args.seed,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        tts_temperature=args.tts_temperature,
        length_penalty=args.length_penalty,
        listen_prob_scale=args.listen_prob_scale,
        max_speak_tokens=args.max_speak_tokens,
    )
    from voicelab.models.minicpm.runs import execute_run

    print(execute_run(args.output, request, args.input, args.trailing_silence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
