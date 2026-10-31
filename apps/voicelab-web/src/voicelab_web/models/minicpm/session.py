"""Bounded browser transport with ownership through native process cleanup."""

import asyncio
import contextlib
import logging
import threading
from dataclasses import asdict
from functools import partial
from typing import Literal

from anyio import CancelScope
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, ValidationError

from voicelab.runs.worker import InferenceWorker
from voicelab_inference.minicpm.native.protocol import Event
from voicelab_inference.minicpm.native.runtime import MiniCPMNative
from voicelab_models.minicpm.model import FRAME_BYTES, MAX_SECONDS, Request

logger = logging.getLogger(__name__)


class Start(Request):
    type: Literal["start"]


class Stop(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    type: Literal["stop"]


async def serve_session(socket: WebSocket) -> None:
    await socket.accept()
    worker = InferenceWorker("minicpm")
    runtime = MiniCPMNative()
    cancelled = threading.Event()
    ready = False
    tasks: list[asyncio.Task[None]] = []
    try:
        raw = await asyncio.wait_for(socket.receive_text(), 30)
        if len(raw) > 9 * 1024 * 1024:
            raise ValueError("MiniCPM request exceeds 9 MiB.")
        start = Start.model_validate_json(raw)
        request = Request.model_validate(start.model_dump(exclude={"type"}))
        frames: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=4)
        loop = asyncio.get_running_loop()

        async def deliver(event: Event) -> None:
            if event.kind == "audio":
                await socket.send_bytes(event.pcm)
            elif event.kind == "text":
                await socket.send_json(
                    {"type": "assistant_text_delta", "text": event.text}
                )
            else:
                await socket.send_json({"type": "listen"})

        def emit(event: Event) -> None:
            if cancelled.is_set():
                raise InterruptedError("MiniCPM session cancelled.")
            future = asyncio.run_coroutine_threadsafe(deliver(event), loop)
            try:
                future.result(timeout=5)
            except BaseException:
                future.cancel()
                cancelled.set()
                raise

        async def receive() -> None:
            count = 0
            try:
                while True:
                    message = await socket.receive()
                    if message["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect()
                    pcm = message.get("bytes")
                    if pcm is None:
                        Stop.model_validate_json(message.get("text") or "")
                        if not ready or request.mode == "turn_based":
                            cancelled.set()
                        await frames.put(None)
                        return
                    if (
                        not ready
                        or request.mode != "full_duplex"
                        or len(pcm) != FRAME_BYTES
                    ):
                        raise ValueError(
                            "Send one-second PCM16 frames only after duplex is ready."
                        )
                    count += 1
                    if count > MAX_SECONDS:
                        raise ValueError("MiniCPM session exceeds 120 seconds.")
                    try:
                        frames.put_nowait(pcm)
                    except asyncio.QueueFull as exc:
                        raise ValueError(
                            "MiniCPM input fell more than four seconds behind."
                        ) from exc
            except BaseException:
                cancelled.set()
                raise

        async def generate() -> None:
            nonlocal ready
            await socket.send_json({"type": "loading"})
            await worker.call(partial(runtime.start, request, cancelled.is_set))
            ready = True
            await socket.send_json(
                {
                    "type": "ready",
                    "input_sample_rate": 16000,
                    "output_sample_rate": 24000,
                    "setup_seconds": runtime.setup_seconds,
                }
            )
            if request.mode == "turn_based":
                await worker.call(partial(runtime.turn, emit, cancelled.is_set))
            else:
                async with asyncio.timeout(150):
                    while (pcm := await frames.get()) is not None:
                        await worker.call(
                            partial(runtime.push, pcm, emit, cancelled.is_set)
                        )
                        await socket.send_json(
                            {"type": "metrics", **asdict(runtime.metrics())}
                        )
            await worker.call(partial(runtime.finish, emit, cancelled.is_set))
            await socket.send_json(
                {"type": "done", "metrics": asdict(runtime.metrics())}
            )

        receiving = asyncio.create_task(receive())
        generating = asyncio.create_task(generate())
        tasks = [receiving, generating]
        async with asyncio.timeout(1800):
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if receiving in done:
                receiving.result()
                await generating
            else:
                generating.result()
    except WebSocketDisconnect, asyncio.CancelledError, InterruptedError:
        pass
    except ValidationError as exc:
        errors = exc.errors(
            include_input=False, include_url=False, include_context=False
        )
        message = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in errors
        )
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await socket.send_json({"type": "error", "message": message})
    except (ValueError, RuntimeError, TimeoutError) as exc:
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await socket.send_json({"type": "error", "message": str(exc)})
    except Exception:
        logger.exception("MiniCPM local inference failed")
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await socket.send_json(
                {"type": "error", "message": "MiniCPM local inference failed."}
            )
    finally:
        cancelled.set()
        with CancelScope(shield=True):
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await worker.call(runtime.close)
            finally:
                worker.shutdown()
                with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                    await socket.close()
