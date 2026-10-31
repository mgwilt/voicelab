"""Local turn-based Ming Audio with bounded input and cooperative cancellation."""

import asyncio
import contextlib
import logging
import threading
import time
from dataclasses import asdict
from functools import partial
from typing import Literal

from anyio import CancelScope
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, ValidationError

from voicelab_inference.ming_audio.mlx.runtime import MingAudioMLX
from voicelab_models.ming_audio.model import Request
from voicelab_web.transport.worker import InferenceWorker

logger = logging.getLogger(__name__)


class StartMessage(Request):
    type: Literal["start"]


class StopMessage(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    type: Literal["stop"]


async def serve_session(socket: WebSocket) -> None:
    await socket.accept()
    worker = InferenceWorker("ming-audio")
    runtime = MingAudioMLX()
    cancelled = threading.Event()
    try:
        raw = await asyncio.wait_for(socket.receive_text(), 30)
        if len(raw) > 6 * 1024 * 1024:
            raise ValueError("Ming Audio request exceeds its size limit.")
        start = StartMessage.model_validate_json(raw)
        request = Request.model_validate(start.model_dump(exclude={"type"}))
        await socket.send_json({"type": "loading"})

        async def wait_stop() -> None:
            try:
                message = await socket.receive()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect()
                StopMessage.model_validate_json(message.get("text") or "")
            finally:
                cancelled.set()

        async def generate() -> None:
            beginning = time.perf_counter()
            await worker.call(runtime.load)
            if cancelled.is_set():
                return
            await socket.send_json(
                {
                    "type": "ready",
                    "setup_seconds": time.perf_counter() - beginning,
                    "input_sample_rate": 16000,
                    "output_sample_rate": 44100,
                }
            )
            response = await worker.call(
                partial(runtime.respond, request, cancelled.is_set)
            )
            if response.reasoning is not None:
                await socket.send_json(
                    {
                        "type": "assistant_text_delta",
                        "text": f"Reasoning:\n{response.reasoning}\n\nAnswer:\n",
                    }
                )
            await socket.send_json(
                {"type": "assistant_text_delta", "text": response.text}
            )
            for index in range(0, len(response.pcm), 32768):
                await socket.send_bytes(response.pcm[index : index + 32768])
            await socket.send_json(
                {"type": "done", "metrics": asdict(response.metrics)}
            )

        generation = asyncio.create_task(generate())
        stopping = asyncio.create_task(wait_stop())
        try:
            async with asyncio.timeout(180):
                done, _ = await asyncio.wait(
                    (generation, stopping), return_when=asyncio.FIRST_COMPLETED
                )
                if stopping in done:
                    stopping.result()
                    await socket.send_json(
                        {"type": "error", "message": "Request cancelled."}
                    )
                    return
                generation.result()
        finally:
            cancelled.set()
            generation.cancel()
            stopping.cancel()
            with CancelScope(shield=True):
                await asyncio.gather(generation, stopping, return_exceptions=True)
    except WebSocketDisconnect, asyncio.CancelledError:
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
        logger.exception("Local Ming Audio inference failed")
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await socket.send_json(
                {"type": "error", "message": "Ming Audio inference failed."}
            )
    finally:
        cancelled.set()
        with CancelScope(shield=True):
            try:
                await worker.call(runtime.close)
            finally:
                worker.shutdown()
                with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                    await socket.close()
