"""Replay a bounded real-time input and retain received audio incrementally."""

import hashlib
import math
import time
import wave
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from voicelab_inference.personaplex.mlx.runtime import (
    Metrics,
    PersonaPlexMLX,
)
from voicelab_models.personaplex.model import (
    FRAME_BYTES,
    MAX_FRAMES,
    REVISION,
    SAMPLE_RATE,
    SOURCE_MODEL,
    SOURCE_REVISION,
    Request,
)


class Report(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    schema_version: int = 1
    model: str = SOURCE_MODEL
    checkpoint_revision: str = REVISION
    implementation_source_revision: str = SOURCE_REVISION
    runtime: str = "personaplex/mlx"
    status: Literal["running", "stopped", "failed"] = "running"
    request: Request
    reference_sha256: str | None = None
    input_sha256: str
    trailing_silence_seconds: float
    setup_seconds: float | None = None
    metrics: Metrics | None = None
    text: str = ""
    error: str | None = None
    timing_note: str = (
        "Continuous duplex capture without an utterance EOS. Input RTF is "
        "synchronized model and codec processing divided by input duration. "
        "Loading, prompt setup, file writing, and real-time pacing are excluded."
    )


def read_input(source: Path, trailing_silence: float) -> bytes:
    if not math.isfinite(trailing_silence) or not 0 <= trailing_silence <= 10:
        raise ValueError("Trailing silence must be between 0 and 10 seconds.")
    with wave.open(str(source), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (
            1,
            2,
            SAMPLE_RATE,
        ):
            raise ValueError("PersonaPlex input must be mono PCM16 WAV24k.")
        count = audio.getnframes()
        if not 0 < count <= int((120 - trailing_silence) * SAMPLE_RATE):
            raise ValueError("Input and trailing silence must fit within 120 seconds.")
        pcm = audio.readframes(count)
        if len(pcm) != count * 2:
            raise ValueError("Input WAV is truncated.")
    pcm += bytes(int(trailing_silence * SAMPLE_RATE) * 2)
    pcm += bytes((-len(pcm)) % FRAME_BYTES)
    if len(pcm) > MAX_FRAMES * FRAME_BYTES:
        raise ValueError("Padded input exceeds the 120-second session limit.")
    return pcm


def execute_run(
    source: Path, output: Path, request: Request, trailing_silence: float
) -> Path:
    pcm = read_input(source, trailing_silence)
    backend = PersonaPlexMLX()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "run.json"
    report = Report(
        request=request.model_copy(update={"reference_audio": None}),
        reference_sha256=hashlib.sha256(request.reference_audio.wav_bytes()).hexdigest()
        if request.reference_audio is not None
        else None,
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        trailing_silence_seconds=trailing_silence,
    )
    report_path.write_text(report.model_dump_json(indent=2) + "\n")
    text_parts: list[str] = []
    try:
        beginning = time.perf_counter()
        backend.start(request)
        report = report.model_copy(
            update={"setup_seconds": time.perf_counter() - beginning}
        )
        with wave.open(str(output / "response.wav"), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(SAMPLE_RATE)
            for index in range(0, len(pcm), FRAME_BYTES):
                for event in backend.push(pcm[index : index + FRAME_BYTES]):
                    if event.kind == "audio":
                        audio.writeframesraw(event.pcm)
                    else:
                        text_parts.append(event.text)
        if backend.metrics().output_seconds == 0:
            raise RuntimeError("PersonaPlex returned no audio before capture ended.")
        report = report.model_copy(update={"status": "stopped"})
    except BaseException as exc:
        report = report.model_copy(update={"status": "failed", "error": str(exc)})
        raise
    finally:
        report = report.model_copy(
            update={"metrics": backend.metrics(), "text": "".join(text_parts)}
        )
        try:
            backend.close()
        finally:
            report_path.write_text(report.model_dump_json(indent=2) + "\n")
    return report_path
