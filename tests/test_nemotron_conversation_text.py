"""Nemotron's cumulative transcript stays explicit across the application boundary."""

import unittest

from voicelab.conversations.adapters.nemotron import NemotronDuplex
from voicelab.conversations.contracts import Audio, Notice
from voicelab_models.nemotron.streaming import StreamEvent


class Stream:
    def push(self, pcm: bytes) -> tuple[StreamEvent, ...]:
        return (
            StreamEvent("user_transcript_delta", text="Who"),
            StreamEvent("user_transcript_delta", text="Who are you?"),
            StreamEvent("assistant_text_delta", text="I"),
            StreamEvent("assistant_text_delta", text="I am"),
            StreamEvent("assistant_text_delta", text="I am a language model."),
            StreamEvent("audio", pcm=b"\x01\x00\x02\x00"),
        )

    def close(self) -> None:
        pass


class NemotronTranscriptTests(unittest.TestCase):
    def test_snapshots_remain_partial_and_audio_is_unchanged(self) -> None:
        adapter = NemotronDuplex("Answer briefly.", 0)
        adapter.session = Stream()
        events: list[Audio | Notice] = []
        adapter.process(bytes(2560), events.append, lambda: False)
        text = [
            event
            for event in events
            if isinstance(event, Notice) and event.type == "text"
        ]
        self.assertEqual([event.text_mode for event in text], ["snapshot"] * 5)
        self.assertTrue(all(not event.final for event in text))
        self.assertEqual(
            [event.speaker for event in text], ["user"] * 2 + ["assistant"] * 3
        )
        self.assertEqual(text[-1].text, "I am a language model.")
        self.assertEqual(text[-1].model_dump()["text_mode"], "snapshot")
        audio = [event for event in events if isinstance(event, Audio)]
        self.assertEqual(audio, [Audio(b"\x01\x00\x02\x00", 22050)])
        adapter.close()
