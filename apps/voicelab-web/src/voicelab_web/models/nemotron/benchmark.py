"""Paced WAV replay through the same WebSocket path as the mobile client."""

import argparse
import asyncio
import hashlib
import json
import ssl
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from websockets.asyncio.client import connect

from voicelab_core.audio import inspect_input
from voicelab_models.nemotron.streaming import FRAME_BYTES, FRAME_SECONDS


class ServerMessage(BaseModel):
    # Ignore display-only fields while validating measurements used by the runner.
    model_config = ConfigDict(strict=True, extra="ignore")
    type: str
    message: str = ""
    text: str = ""
    metrics: dict[str, float | int | None] | None = None
    setup_seconds: float | None = None
    backlog_seconds: float = 0.0
    provenance: dict[str, object] | None = None
    checkpoint: dict[str, str] | None = None
    settings: dict[str, str | int] | None = None
    backend: str | None = None


async def trial(
    url: str, context: ssl.SSLContext | None, frames: list[bytes], output: Path
) -> dict[str, object]:
    audio = bytearray()
    transcript = ""
    response = ""
    peak_backlog = 0.0
    metrics: dict[str, float | int | None] | None = None
    async with connect(url, ssl=context, max_size=65536) as socket:
        await socket.send(json.dumps({"type": "start", "seed": 0}))
        while True:
            event = ServerMessage.model_validate_json(await socket.recv())
            if event.type == "error":
                raise RuntimeError(event.message)
            if event.type == "ready":
                setup_seconds = event.setup_seconds
                provenance = event.provenance
                checkpoint = event.checkpoint
                settings = event.settings
                backend = event.backend
                break

        async def send() -> None:
            clock = asyncio.get_running_loop()
            start = clock.time()
            for index, frame in enumerate(frames):
                await asyncio.sleep(
                    max(0, start + index * FRAME_SECONDS - clock.time())
                )
                await socket.send(frame)
            await socket.send('{"type":"stop"}')

        async def receive() -> None:
            nonlocal transcript, response, peak_backlog, metrics
            async for raw in socket:
                if isinstance(raw, bytes):
                    audio.extend(raw)
                    continue
                event = ServerMessage.model_validate_json(raw)
                if event.type == "error":
                    raise RuntimeError(event.message)
                if event.type == "user_transcript_delta":
                    transcript = event.text
                elif event.type == "assistant_text_delta":
                    response = event.text
                elif event.type == "metrics":
                    peak_backlog = max(peak_backlog, event.backlog_seconds)
                elif event.type == "done":
                    metrics = event.metrics
                    return
            raise RuntimeError("Session closed without a completed measurement.")

        async with asyncio.TaskGroup() as group:
            group.create_task(send())
            group.create_task(receive())
    if metrics is None:
        raise RuntimeError("No metrics received.")
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(audio)
    return {
        "metrics": metrics,
        "provenance": provenance,
        "checkpoint": checkpoint,
        "settings": settings,
        "backend": backend,
        "setup_seconds": setup_seconds,
        "peak_backlog_seconds": peak_backlog,
        "text": response,
        "user_transcript": transcript,
        "output_frames": len(audio) // 2,
    }


class Arguments(argparse.Namespace):
    url: str
    ca: Path | None
    input: Path
    output: Path
    trials: int
    silence: float


async def run(args: Arguments) -> None:
    inspect_input(args.input)
    with wave.open(str(args.input), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
    pcm += b"\0\0" * round(args.silence * 16000)
    pcm += b"\0" * (-len(pcm) % FRAME_BYTES)
    frames = [pcm[i : i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]
    context = (
        ssl.create_default_context(cafile=str(args.ca) if args.ca is not None else None)
        if args.url.startswith("wss:")
        else None
    )
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, object] = {
        "started_at": datetime.now(UTC).isoformat(),
        "url": args.url,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "input_frames": len(frames),
        "trailing_silence_seconds": args.silence,
        "notes": (
            "Paced 80 ms frames; fresh session each trial. Model cached after "
            "first load. Cold frames included. No optimization. "
            "Not a microphone or mobile acceptance test."
        ),
    }
    results: list[dict[str, object]] = []
    status: Literal["failed", "completed"] = "failed"
    try:
        for index in range(args.trials):
            result = await trial(
                args.url, context, frames, args.output / f"reply-{index}.wav"
            )
            results.append(result)
            print(json.dumps(result), flush=True)
        status = "completed"
    finally:
        report.update({"status": status, "trials": results})
        (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8443/api/nemotron/session")
    parser.add_argument("--ca", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--silence", type=float, default=5.0)
    args = parser.parse_args(namespace=Arguments())
    if not 1 <= args.trials <= 20 or not 0 <= args.silence <= 30:
        parser.error("Use 1-20 trials and 0-30 seconds trailing silence.")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
