"""Local PersonaPlex duplex inference with bounded browser input."""

import asyncio
import contextlib
import logging
import time
from dataclasses import asdict
from functools import partial
from typing import Literal

from anyio import CancelScope
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, ValidationError

from voicelab_inference.personaplex.mlx.runtime import PersonaPlexMLX
from voicelab_models.personaplex.model import FRAME_BYTES, MAX_FRAMES, Request
from voicelab_web.transport.worker import InferenceWorker

logger = logging.getLogger(__name__)


class StartMessage(Request):
    type: Literal["start"]


class StopMessage(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    type: Literal["stop"]


async def serve_session(socket: WebSocket) -> None:
    """Run under the application lease; release only after local cleanup."""
    await socket.accept()
    worker = InferenceWorker("personaplex")
    runtime = PersonaPlexMLX()
    tasks: list[asyncio.Task[None]] = []
    try:
        raw = await asyncio.wait_for(socket.receive_text(), 15)
        if len(raw) > 6 * 1024 * 1024:
            raise ValueError("PersonaPlex start message exceeds 6 MiB.")
        start = StartMessage.model_validate_json(raw)
        request = Request.model_validate(start.model_dump(exclude={"type"}))
        await socket.send_json({"type": "loading"})
        beginning = time.perf_counter()
        await worker.call(partial(runtime.start, request))
        await socket.send_json(
            {
                "type": "ready",
                "setup_seconds": time.perf_counter() - beginning,
                "input_sample_rate": 24000,
                "output_sample_rate": 24000,
            }
        )
        frames: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=25)

        async def receive() -> None:
            count = 0
            while True:
                message = await asyncio.wait_for(socket.receive(), 30)
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect()
                pcm = message.get("bytes")
                if pcm is None:
                    StopMessage.model_validate_json(message.get("text") or "")
                    await frames.put(None)
                    return
                if len(pcm) != FRAME_BYTES:
                    raise ValueError("Send 1920 mono PCM16 samples per frame.")
                count += 1
                if count > MAX_FRAMES:
                    raise ValueError("PersonaPlex input exceeds 120 seconds.")
                try:
                    frames.put_nowait(pcm)
                except asyncio.QueueFull as exc:
                    raise ValueError(
                        "PersonaPlex input fell 2 seconds behind."
                    ) from exc

        async def process() -> None:
            while True:
                pcm = await frames.get()
                if pcm is None:
                    await socket.send_json(
                        {"type": "done", "metrics": asdict(runtime.metrics())}
                    )
                    return
                events = await worker.call(partial(runtime.push, pcm))
                for event in events:
                    if event.kind == "audio":
                        await socket.send_bytes(event.pcm)
                    else:
                        await socket.send_json(
                            {"type": "assistant_text_delta", "text": event.text}
                        )
                await socket.send_json({"type": "metrics", **asdict(runtime.metrics())})

        tasks = [asyncio.create_task(receive()), asyncio.create_task(process())]
        async with asyncio.timeout(180):
            await asyncio.gather(*tasks)
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
        logger.exception("Local PersonaPlex session failed")
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await socket.send_json(
                {"type": "error", "message": "PersonaPlex inference failed."}
            )
    finally:
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
