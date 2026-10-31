"""Keep reasoning and language labels separate from the spoken answer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Answer:
    text: str
    reasoning: str | None
    language: str | None
    reasoning_complete: bool


def parse_answer(value: object, *, thinking: bool, transcription: bool) -> Answer:
    if not isinstance(value, str):
        raise TypeError("Ming tokenizer must return text")
    text = value.strip()
    reasoning: str | None = None
    language: str | None = None
    reasoning_complete = True
    if thinking or text.startswith("<think>"):
        reasoning, separator, answer = text.removeprefix("<think>").partition(
            "</think>"
        )
        reasoning = reasoning.strip()
        text = answer.strip() if separator else ""
        reasoning_complete = bool(separator)
    if transcription:
        if "\t" in text:
            language, text = text.split("\t", 1)
            language, text = language.strip(), text.strip()
        else:
            for label in ("Chinese", "English", "Canton", "Japanese", "Korean"):
                if text.startswith(label):
                    language, text = label, text[len(label) :].strip()
                    break
    return Answer(text, reasoning, language, reasoning_complete)
