"""Greedy two-channel decoding with response-only teacher forcing. See NOTICE."""

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
from mlx_vlm.models.nemotron_voicechat.session import VoiceChatSession

from voicelab_inference.nemotron.mlx.sampling import preserved
from voicelab_inference.nemotron.mlx.tool_timeline import AudioTimeline
from voicelab_models.nemotron.tools import ToolTimeline


@dataclass(frozen=True)
class DecodedTurn:
    text_tokens: tuple[int, ...]
    function_tokens: tuple[int, ...]
    audio: mx.array | None


def decode(
    session: VoiceChatSession,
    timeline: AudioTimeline,
    *,
    seed: int,
    synthesize: bool,
    plan: ToolTimeline | None,
    should_cancel: Callable[[], bool],
) -> DecodedTurn:
    with preserved():
        return _decode(
            session,
            timeline,
            seed=seed,
            synthesize=synthesize,
            plan=plan,
            should_cancel=should_cancel,
        )


def _decode(
    session: VoiceChatSession,
    timeline: AudioTimeline,
    *,
    seed: int,
    synthesize: bool,
    plan: ToolTimeline | None,
    should_cancel: Callable[[], bool],
) -> DecodedTurn:
    model = session.model
    mx.random.seed(seed)
    frames = timeline.audio.shape[1]
    pad = model.config.pad_token_id
    texts = [pad] * frames
    functions = [pad] * frames
    cache = model.stt_model.make_cache()
    # EAR-TTS cache/code state is independent of the language-model cache.
    prompt = session._tts_prompt() if synthesize else None
    tts_cache = None
    previous_code = None
    if prompt is not None:
        _, tts_cache = model.tts_model.tts_model.warmup(
            *prompt[:-1], audio_prompt_latent=prompt[-1], guidance=True
        )
        previous_code = prompt[0][:, -1:]
    codes = [mx.zeros((1, 1, model.config.tts_config.num_quantizers), dtype=mx.int32)]
    for time in range(frames):
        if should_cancel():
            raise InterruptedError("Nemotron tool experiment cancelled.")
        previous_text = mx.array(
            [[pad if time == 0 else texts[time - 1]]], dtype=mx.int32
        )
        previous_function = mx.array(
            [[pad if time == 0 else functions[time - 1]]], dtype=mx.int32
        )
        fused = (
            model.stt_model.embed_tokens(previous_text)
            + timeline.audio[:, time : time + 1]
            + model.config.function_channel_weight
            * model.stt_model.embed_tokens(previous_function)
        )
        output = model.stt_model(fused, cache=cache)
        if time >= timeline.prompt_frames:
            texts[time] = int(mx.argmax(output.text_logits[:, -1]))
            forced = plan.forced_token(time - timeline.prompt_frames) if plan else None
            # NVIDIA treats PAD as unfilled even within a response span.
            functions[time] = (
                forced
                if forced is not None and forced != pad
                else int(mx.argmax(output.function_logits[:, -1]))
            )
        if time == 0 or not synthesize:
            continue
        if previous_code is None or tts_cache is None:
            raise RuntimeError("Missing EAR-TTS state.")
        current = mx.array([[texts[time]]], dtype=mx.int32)
        if texts[time] == model.config.eos_token_id:
            previous_code = mx.broadcast_to(
                model.tts_model.codec_silence_tokens[None, None, :], previous_code.shape
            )
        tts_output = model.tts_model.tts_model.step(
            previous_code,
            current,
            mx.ones(current.shape, dtype=mx.bool_),
            tts_cache,
            guidance=True,
        )
        previous_code = tts_output.codes
        codes.append(previous_code)
        mx.eval(previous_code, output.text_logits, output.function_logits)
    audio = None
    if synthesize:
        generated = mx.concatenate(codes, axis=1)[:, timeline.prompt_frames :]
        generated = session._replace_control_codes(generated)
        audio = model.tts_model.audio_codec.decode(generated.transpose(0, 2, 1))[0, 0]
        mx.eval(audio)
    mx.synchronize()
    return DecodedTurn(
        tuple(texts[timeline.prompt_frames :]),
        tuple(functions[timeline.prompt_frames :]),
        audio,
    )
