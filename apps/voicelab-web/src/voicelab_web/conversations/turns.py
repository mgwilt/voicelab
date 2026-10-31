"""Local speech boundaries, bounded pre-roll, and semantic end-of-turn detection."""

from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADState


class TurnDetector:
    def __init__(self) -> None:
        self.vad = SileroVADAnalyzer(sample_rate=16000)
        self.vad.set_sample_rate(16000)
        self.smart = LocalSmartTurnAnalyzerV3(sample_rate=16000)
        self.smart.set_sample_rate(16000)
        self.audio = bytearray()
        self.speaking = False
        self.active = False

    async def push(self, pcm: bytes) -> tuple[bool, bytes | None]:
        state = await self.vad.analyze_audio(pcm)
        speaking = state in (VADState.SPEAKING, VADState.STOPPING)
        began = speaking and not self.speaking
        complete = self.smart.append_audio(pcm, speaking) == EndOfTurnState.COMPLETE
        if self.speaking and not speaking:
            end, _ = await self.smart.analyze_end_of_turn()
            complete = complete or end == EndOfTurnState.COMPLETE
        self.speaking = speaking
        self.active = self.active or speaking
        self.audio.extend(pcm)
        if not self.active:
            del self.audio[:-16000]  # Half a second of pre-roll.
        elif len(self.audio) > 16000 * 2 * 30:
            raise ValueError("A spoken turn exceeds 30 seconds")
        if complete and self.active:
            turn = bytes(self.audio)
            self.audio.clear()
            self.active = False
            self.smart.clear()
            return began, turn
        return began, None

    async def close(self) -> None:
        await self.vad.cleanup()
        await self.smart.cleanup()
