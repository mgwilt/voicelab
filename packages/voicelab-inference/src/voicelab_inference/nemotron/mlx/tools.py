"""Local, supplied-response execution of NVIDIA's two-pass offline workflow."""

import time
import wave
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm.models.nemotron_voicechat.session import VoiceChatSession

from voicelab_core.contracts import GenerationSettings
from voicelab_inference.nemotron.mlx.tool_calls import detect_call
from voicelab_inference.nemotron.mlx.tool_decode import decode
from voicelab_inference.nemotron.mlx.tool_result import ToolResult
from voicelab_inference.nemotron.mlx.tool_timeline import expand, prepare
from voicelab_models.nemotron.model import OUTPUT_SAMPLE_RATE
from voicelab_models.nemotron.tools import ToolSettings, ToolTimeline


def never_cancel() -> bool:
    return False


def generate_with_response(
    session: VoiceChatSession,
    source: Path,
    destination: Path,
    settings: GenerationSettings,
    tools: ToolSettings,
    should_cancel: Callable[[], bool] = never_cancel,
) -> ToolResult:
    started = time.perf_counter()
    timeline = prepare(
        session,
        source,
        tools.system_prompt(settings.prompt),
        settings.trailing_silence_seconds,
    )
    first = decode(
        session,
        timeline,
        seed=settings.seed,
        synthesize=False,
        plan=None,
        should_cancel=should_cancel,
    )
    call = detect_call(first.function_tokens, session.tokenizer)
    if call.call.name != tools.api_response.tool_name:
        raise ValueError("Predicted tool name does not match the supplied response.")
    response_ids = tuple(
        session.tokenizer.encode(tools.response_text(), add_special_tokens=False)
    )
    plan = ToolTimeline(call.start_frame, call.token_count, response_ids)
    detection_seconds = time.perf_counter() - started
    started = time.perf_counter()
    expanded = expand(session, timeline, plan)
    second = decode(
        session,
        expanded,
        seed=settings.seed,
        synthesize=True,
        plan=plan,
        should_cancel=should_cancel,
    )
    if second.audio is None:
        raise RuntimeError("Tool continuation returned no audio.")
    samples = np.asarray(second.audio, dtype=np.float32).reshape(-1)
    if not samples.size or not np.isfinite(samples).all():
        raise RuntimeError("Tool continuation produced empty or non-finite audio.")
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(destination), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(OUTPUT_SAMPLE_RATE)
        stream.writeframes(pcm.tobytes())
    response_end = plan.response_start + len(response_ids)
    eotr = session.tokenizer.encode("<SPECIAL_22>", add_special_tokens=False)
    if len(eotr) != 1:
        raise RuntimeError("Nemotron EOTR marker must be a single token.")
    return ToolResult(
        call,
        session._decode_text(mx.array(second.text_tokens, dtype=mx.int32)),
        session._decode_text(
            mx.array(second.text_tokens[response_end + 1 :], dtype=mx.int32)
        ),
        second.function_tokens,
        plan.response_start,
        response_end,
        second.function_tokens[response_end] == eotr[0],
        detection_seconds,
        time.perf_counter() - started,
        len(samples) / OUTPUT_SAMPLE_RATE,
    )
