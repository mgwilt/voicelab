"""Stateful Nemotron MLX streaming; one owner thread per loaded engine."""

from typing import TYPE_CHECKING

from voicelab_models.nemotron.streaming import FRAME_BYTES, StreamEvent

if TYPE_CHECKING:
    from mlx_vlm.models.nemotron_voicechat import VoiceChatSession


class NemotronStream:
    def __init__(self, parent: VoiceChatSession, prompt: str, seed: int) -> None:
        if not prompt.isascii():
            raise ValueError("Nemotron system prompts must be ASCII.")
        self._stream = parent.create_streaming_session(
            system_prompt=prompt,
            seed=seed,
            profile=True,
            max_streaming_seconds=120,
        )

    def push(self, pcm: bytes) -> tuple[StreamEvent, ...]:
        import mlx.core as mx
        import numpy as np

        if len(pcm) != FRAME_BYTES:
            raise ValueError("Expected exactly 80 ms of mono 16 kHz PCM16.")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        events = self._stream.push_audio(samples, sample_rate=16000)
        result: list[StreamEvent] = []
        for event in events:
            if event.kind == "audio" and event.samples is not None:
                mx.eval(event.samples)
                audio = np.asarray(event.samples, dtype=np.float32).reshape(-1)
                if not np.isfinite(audio).all():
                    raise RuntimeError("Non-finite model audio.")
                pcm_out = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                result.append(
                    StreamEvent("audio", pcm=pcm_out, frame_index=event.frame_index)
                )
            elif event.kind in {
                "assistant_text_delta",
                "user_transcript_delta",
                "function_delta",
            }:
                # Keep the kind narrowed to the supported text event literals.
                kind = event.kind
                if (
                    kind == "assistant_text_delta"
                    or kind == "user_transcript_delta"
                    or kind == "function_delta"
                ):
                    result.append(StreamEvent(kind, text=event.text or ""))
        mx.synchronize()
        return tuple(result)

    def close(self) -> None:
        import mlx.core as mx

        self._stream.cancel()
        mx.synchronize()
