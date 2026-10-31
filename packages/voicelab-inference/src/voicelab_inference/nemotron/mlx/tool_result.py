"""Device-independent evidence from a Nemotron tool-response experiment."""

from dataclasses import dataclass

from voicelab_models.nemotron.tools import DetectedCall


@dataclass(frozen=True)
class ToolResult:
    call: DetectedCall
    text: str
    continuation_text: str
    function_tokens: tuple[int, ...]
    response_start_frame: int
    response_end_frame: int
    predicted_end_response: bool
    detection_seconds: float
    continuation_seconds: float
    output_seconds: float

    @property
    def completed(self) -> bool:
        """The supplied response ended and text continued; no utterance-EOS claim."""
        return self.predicted_end_response and bool(self.continuation_text.strip())
