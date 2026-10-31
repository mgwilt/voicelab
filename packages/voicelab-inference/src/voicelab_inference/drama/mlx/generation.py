"""Native Drama orchestration with explicit prior text/audio continuation turns.

Adapted from MLX-Audio 0.6.3 Model.generate; see LICENSE and NOTICE.
Model math, frame sampling, codec decoding, and text chunking stay upstream.
"""

import time
from collections.abc import Generator
from contextlib import closing

import mlx.core as mx
from mlx_audio.codec.models.fish_s1_dac import DAC
from mlx_audio.tts.models.base import GenerationResult
from mlx_audio.tts.models.fish_drama3.chunking import split_text
from mlx_audio.tts.models.fish_drama3.drama3 import AudioInput, Model

from voicelab_models.drama.model import SAMPLE_RATE, Request


def generate(
    model: Model,
    codec: DAC,
    request: Request,
    references: list[AudioInput],
    previous_takes: list[tuple[str, mx.array]],
) -> Generator[GenerationResult]:
    voice = model._voice(references, [item.text for item in request.references])
    interval = max(
        1, round(request.streaming_interval * SAMPLE_RATE / codec.frame_length)
    )
    first = request.first_chunk_frames or interval
    chunks = split_text(request.text)
    pinned: tuple[str, mx.array] | None = None
    latest: tuple[str, mx.array] | None = None
    started = time.perf_counter()
    codec.reset_streaming_state()
    try:
        for index, text in enumerate(chunks):
            final = index == len(chunks) - 1
            # Prior takes are full teacher-forced turns, separate from voice references.
            history = [
                *previous_takes,
                *(turn for turn in (pinned, latest) if turn is not None),
            ]
            prompt, system_rows = model.build_frames(voice.references, history, text)
            codes: list[mx.array] = []
            emitted = 0
            frames = model._frames(
                prompt,
                system_rows,
                voice,
                request.max_new_tokens,
                request.temperature,
                request.top_k,
                request.top_p,
            )
            if not isinstance(frames, Generator):
                raise TypeError("Drama frame decoder must return a closable generator")
            with closing(frames):
                for frame in frames:
                    codes.append(frame)
                    due = first if index == 0 and emitted == 0 else interval
                    if len(codes) - emitted >= due:
                        yield model._result(
                            codes[emitted:], index, started, True, False
                        )
                        emitted = len(codes)
                        started = time.perf_counter()
            if codes:
                turn = (text, mx.stack(codes, axis=1))
                if pinned is None:
                    pinned = turn
                else:
                    latest = turn
            if len(codes) > emitted or final:
                result = model._result(codes[emitted:], index, started, True, final)
                mx.clear_cache()
                yield result
                started = time.perf_counter()
            else:
                mx.clear_cache()
    finally:
        mx.clear_cache()
