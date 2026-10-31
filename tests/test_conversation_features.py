"""Task boundaries, conditioning validation, and text-only conversation metrics."""

import base64
import unittest
from collections.abc import Callable

from pydantic import ValidationError
from tests.test_conversation_adapters import wav_bytes

from voicelab_inference.ming_audio.mlx.answer import parse_answer
from voicelab_inference.ming_audio.mlx.contracts import Output
from voicelab_inference.ming_audio.mlx.runtime import MingAudioMLX
from voicelab_inference.personaplex.mlx.voice import reference_wave
from voicelab_models.audio import ReferenceAudio
from voicelab_models.ming_audio.model import HistoryTurn, Request
from voicelab_models.personaplex.model import Request as PersonaRequest


def reference() -> ReferenceAudio:
    return ReferenceAudio(wav_base64=base64.b64encode(wav_bytes(frames=16000)).decode())


class ConversationFeatureTests(unittest.TestCase):
    def test_task_requires_appropriate_input(self) -> None:
        for task in ("asr", "audio_understanding"):
            with self.assertRaises(ValidationError):
                Request.model_validate({"task": task, "prompt": "Transcribe"})
        with self.assertRaises(ValidationError):
            Request(task="tts", audio_wav_base64=reference().wav_base64)
        with self.assertRaises(ValidationError):
            Request(task="text", reference_audio=reference())
        with self.assertRaises(ValidationError):
            Request(task="tts", reference_text="An orphaned transcript")

    def test_history_preserves_audio_and_alternating_turns(self) -> None:
        history = (
            HistoryTurn(role="user", audio=reference()),
            HistoryTurn(role="assistant", content="Previous answer"),
        )
        request = Request(task="text", prompt="What was the sound?", history=history)
        self.assertEqual(request.history[0].audio, reference())
        with self.assertRaises(ValidationError):
            Request(task="text", history=(history[1],))
        with self.assertRaises(ValidationError):
            Request(task="text", history=(history[0],))

    def test_incomplete_reasoning_is_not_spoken_as_an_answer(self) -> None:
        answer = parse_answer(
            "<think>unfinished thought", thinking=True, transcription=False
        )
        self.assertEqual(answer.text, "")
        self.assertFalse(answer.reasoning_complete)
        self.assertEqual(answer.reasoning, "unfinished thought")
        final = parse_answer(
            "<think>Reason</think>English\tHello", thinking=True, transcription=True
        )
        self.assertEqual(
            (final.text, final.language, final.reasoning),
            ("Hello", "English", "Reason"),
        )
        self.assertTrue(final.reasoning_complete)

    def test_text_only_response_has_no_invented_audio_rtf(self) -> None:
        class Backend:
            def respond(
                self, request: Request, should_cancel: Callable[[], bool]
            ) -> Output:
                return Output("Answer", None, True)

            def close(self) -> None:
                pass

        runtime = MingAudioMLX(factory=Backend)
        result = runtime.respond(Request(task="text", prompt="Say hello"))
        runtime.close()
        self.assertEqual(result.pcm, b"")
        self.assertEqual(result.metrics.output_seconds, 0)
        self.assertIsNone(result.metrics.rtf)
        self.assertIsNone(result.metrics.first_pcm_seconds)
        self.assertTrue(result.metrics.completed)

    def test_persona_reference_rejects_silence_before_model_loading(self) -> None:
        with self.assertRaisesRegex(ValueError, "silence"):
            reference_wave(PersonaRequest(reference_audio=reference()))
        with self.assertRaises(ValidationError):
            PersonaRequest(seed=-1)
        with self.assertRaises(ValidationError):
            PersonaRequest(audio_temperature=float("nan"))


if __name__ == "__main__":
    unittest.main()
