"""Qwen Omni audio/text prompts, history, speakers, and generation controls."""

import argparse
from pathlib import Path

from pydantic import TypeAdapter

from voicelab_models.audio import ReferenceAudio
from voicelab_models.qwen_omni.model import SYSTEM_PROMPT, HistoryTurn, Request, Speaker


class Arguments(argparse.Namespace):
    input: Path | None
    output: Path
    prompt: str
    system_prompt: str
    history: Path | None
    speaker: Speaker
    text_only: bool
    buffered: bool
    seed: int
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    talker_max_tokens: int
    talker_temperature: float
    talker_top_p: float
    talker_top_k: int
    talker_repetition_penalty: float
    chunk_frames: int


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run local Qwen3-Omni Thinker, Talker and codec with MLX."
    )
    parser.add_argument(
        "--input", type=Path, help="Mono PCM16 WAV at 8-48 kHz, up to 30 seconds"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", "--text", default="Answer briefly.")
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument(
        "--history",
        type=Path,
        help="JSON array of prior user/assistant pairs; audio uses ReferenceAudio",
    )
    parser.add_argument(
        "--speaker", choices=("Ethan", "Chelsie", "Aiden"), default="Ethan"
    )
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--buffered", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--talker-max-tokens", type=int, default=1500)
    parser.add_argument("--talker-temperature", type=float, default=0.9)
    parser.add_argument("--talker-top-p", type=float, default=1)
    parser.add_argument("--talker-top-k", type=int, default=50)
    parser.add_argument("--talker-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--chunk-frames", type=int, default=25)
    args = parser.parse_args(namespace=Arguments())
    history: tuple[HistoryTurn, ...] = ()
    if args.history is not None:
        if args.history.stat().st_size > 12 * 1024 * 1024 + 65536:
            parser.error("History exceeds 12 MiB.")
        history = TypeAdapter(tuple[HistoryTurn, ...]).validate_json(
            args.history.read_bytes()
        )
    request = Request(
        audio=ReferenceAudio.from_path(args.input) if args.input else None,
        prompt=args.prompt,
        system_prompt=args.system_prompt,
        history=history,
        speaker=args.speaker,
        return_audio=not args.text_only,
        stream_audio=not args.buffered,
        seed=args.seed,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        talker_max_tokens=args.talker_max_tokens,
        talker_temperature=args.talker_temperature,
        talker_top_p=args.talker_top_p,
        talker_top_k=args.talker_top_k,
        talker_repetition_penalty=args.talker_repetition_penalty,
        chunk_frames=args.chunk_frames,
    )
    from voicelab.models.qwen_omni.runs import execute_run

    print(execute_run(request, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
