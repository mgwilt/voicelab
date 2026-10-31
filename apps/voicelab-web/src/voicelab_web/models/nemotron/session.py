"""Bounded, concurrent WebSocket input and model output for one session."""

import asyncio
import contextlib
import logging
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

from anyio import CancelScope
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from voicelab_core.provenance import runtime_evidence
from voicelab_models.nemotron.model import MLX_8BIT
from voicelab_models.nemotron.streaming import FRAME_BYTES, FRAME_SECONDS
from voicelab_web.models.nemotron.engine import Engine
from voicelab_web.models.nemotron.metrics import FrameMetrics
from voicelab_web.models.nemotron.schemas import StartMessage, StopMessage

logger = logging.getLogger(__name__)


async def serve_session(socket: WebSocket, engine: Engine) -> None:
    # Prevent cross-origin pages from starting inference on a LAN-accessible server.
    origin = socket.headers.get("origin")
    expected = (
        ("https" if socket.url.scheme == "wss" else "http")
        + "://"
        + socket.headers.get("host", "")
    )
    if origin is not None and origin != expected:
        await socket.close(code=1008)
        return
    await socket.accept()
    if engine.busy:
        await socket.send_json(
            {"type": "error", "message": "Another session is active."}
        )
        await socket.close(code=1013)
        return
    engine.busy = True
    tasks: list[asyncio.Task[None]] = []
    try:
        start = StartMessage.model_validate_json(
            await asyncio.wait_for(socket.receive_text(), 15)
        )
        await socket.send_json({"type": "loading"})
        setup_seconds = await engine.call(
            lambda: engine.start(start.prompt, start.seed)
        )
        await socket.send_json(
            {
                "type": "ready",
                "setup_seconds": setup_seconds,
                "checkpoint": asdict(MLX_8BIT),
                "settings": start.model_dump(),
                "backend": "nemotron/mlx-vlm-streaming-reference",
                "provenance": asdict(
                    runtime_evidence(
                        Path.cwd(),
                        ("mlx-vlm", "mlx", "mlx-metal", "mlx-audio", "numpy"),
                    )
                ),
            }
        )
        queue: asyncio.Queue[tuple[bytes, float] | None] = asyncio.Queue(maxsize=25)
        metrics = FrameMetrics()

        async def receive() -> None:
            frames = 0
            while True:
                message = await asyncio.wait_for(socket.receive(), 30)
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect()
                pcm = message.get("bytes")
                if pcm is None:
                    StopMessage.model_validate_json(message.get("text") or "")
                    await queue.put(None)
                    return
                if len(pcm) != FRAME_BYTES:
                    raise ValueError("Send 1280 mono PCM16 samples per frame.")
                frames += 1
                if frames > 1500:
                    raise ValueError(
                        "Session limit reached (120 seconds). Start a new session."
                    )
                try:
                    queue.put_nowait((pcm, time.perf_counter()))
                except asyncio.QueueFull as exc:
                    raise ValueError(
                        "Inference fell behind: 2 seconds queued. Session stopped."
                    ) from exc

        async def process() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    await socket.send_json(
                        {"type": "done", "metrics": metrics.summary()}
                    )
                    return
                pcm, received_at = item
                queue_ms = (time.perf_counter() - received_at) * 1000
                events, elapsed = await engine.call(partial(engine.push, pcm))
                metrics.add(elapsed)
                for event in events:
                    if event.kind == "audio":
                        await socket.send_bytes(event.pcm)
                    else:
                        await socket.send_json({"type": event.kind, "text": event.text})
                await socket.send_json(
                    {
                        "type": "metrics",
                        **metrics.summary(),
                        "queue_ms": queue_ms,
                        "backlog_seconds": queue.qsize() * FRAME_SECONDS,
                    }
                )

        tasks = [asyncio.create_task(receive()), asyncio.create_task(process())]
        async with asyncio.timeout(180):
            await asyncio.gather(*tasks)
    except WebSocketDisconnect, asyncio.CancelledError:
        pass
    except (ValidationError, ValueError, TimeoutError) as exc:
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await socket.send_json({"type": "error", "message": str(exc)})
    except Exception:
        logger.exception("Inference session failed")
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await socket.send_json(
                {
                    "type": "error",
                    "message": "Inference failed; see the local server log.",
                }
            )
    finally:
        with CancelScope(shield=True):
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await engine.call(engine.close_session)
            finally:
                engine.busy = False
                with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                    await socket.close()
