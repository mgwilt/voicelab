"""Conversation protocol validation without model weights or accelerators."""

import base64
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from voicelab.models.ming_audio.runs import execute_run as run_ming
from voicelab.models.personaplex.runs import read_input
from voicelab_inference.ming_audio.mlx.contracts import Metrics, Response
from voicelab_models.ming_audio.model import Request as MingRequest
from voicelab_models.personaplex.model import FRAME_BYTES
from voicelab_models.personaplex.model import Request as PersonaRequest


def wav_bytes(rate: int = 16000, frames: int = 1600) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(bytes(frames * 2))
    return buffer.getvalue()


def request() -> MingRequest:
    return MingRequest(audio_wav_base64=base64.b64encode(wav_bytes()).decode())


class RequestTests(unittest.TestCase):
    def test_voice_cannot_escape_prompt_directory(self) -> None:
        for voice in ("../NATF2.pt", "/tmp/file.pt", "NATF9.pt"):
            with self.assertRaises(ValidationError):
                PersonaRequest(voice=voice)

    def test_request_rejects_malformed_audio_and_unknown_fields(self) -> None:
        for encoded in ("not base64", base64.b64encode(wav_bytes(24000)).decode()):
            with self.assertRaises(ValidationError):
                MingRequest(audio_wav_base64=encoded)
        with self.assertRaises(ValidationError):
            MingRequest.model_validate({**request().model_dump(), "server_url": "bad"})

    def test_input_has_explicit_duration_limit(self) -> None:
        encoded = base64.b64encode(wav_bytes(frames=16000 * 31)).decode()
        with self.assertRaises(ValidationError):
            MingRequest(audio_wav_base64=encoded)

    def test_incomplete_local_response_retains_audio_and_report(self) -> None:
        class Backend:
            load_seconds = 0.5

            def load(self) -> None:
                pass

            def respond(self, value: MingRequest) -> Response:
                return Response(
                    "partial", bytes(8820), Metrics(0.1, 0.1, 0.2, 2.0, 0.2, False)
                )

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch("voicelab.models.ming_audio.runs.MingAudioMLX", Backend),
                self.assertRaisesRegex(RuntimeError, "generation limit"),
            ):
                run_ming(request(), output)
            report = json.loads((output / "run.json").read_text())
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["runtime"], "ming_audio/mlx")
            self.assertFalse(report["metrics"]["completed"])
            self.assertTrue((output / "response.wav").is_file())
            with self.assertRaises(FileExistsError):
                run_ming(request(), output)

    def test_personaplex_file_validation_precedes_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.wav"
            source.write_bytes(wav_bytes())
            with self.assertRaises(ValueError):
                read_input(source, 3.0)
            source.write_bytes(wav_bytes(24000, 2400))
            pcm = read_input(source, 0.0)
            self.assertEqual(len(pcm) % FRAME_BYTES, 0)
            with self.assertRaises(ValueError):
                read_input(source, float("nan"))


if __name__ == "__main__":
    unittest.main()
