"""Local Qwen speech sessions with streamed PCM and retained worker ownership."""

import asyncio
import logging
from contextlib import suppress
from dataclasses import asdict
from typing import Literal

from anyio import CancelScope
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from voicelab.runs.worker import InferenceWorker
from voicelab_inference.qwen_omni.mlx.contracts import Metrics
from voicelab_inference.qwen_omni.mlx.runtime import QwenOmniMLX
from voicelab_models.qwen_omni.model import Request
from voicelab_web.models.qwen_omni.delivery import Delivery, Ready, Text

logger = logging.getLogger(__name__)


class StartMessage(Request):
    type: Literal["start"]


async def serve_session(socket: WebSocket) -> None:
    await socket.accept()
    worker = InferenceWorker("qwen-omni")
    runtime = QwenOmniMLX()
    delivery = Delivery()
    generation: asyncio.Task[None] | None = None
    stopping: asyncio.Task[None] | None = None
    try:
        raw = await asyncio.wait_for(socket.receive_text(), 30)
        if len(raw) > 13 * 1024 * 1024:
            raise ValueError("Qwen request exceeds 13 MiB.")
        message = StartMessage.model_validate_json(raw)
        request = Request.model_validate(message.model_dump(exclude={"type"}))
        await socket.send_json({"type": "loading"})

        def generate() -> None:
            runtime.load()
            delivery.put(Ready(runtime.load_seconds))
            response = runtime.respond(
                request, delivery.text, delivery.audio, delivery.cancelled.is_set
            )
            delivery.put(response.metrics)

        async def stop() -> None:
            try:
                await socket.receive()
            finally:
                delivery.cancelled.set()

        generation = asyncio.create_task(worker.call(generate))
        stopping = asyncio.create_task(stop())
        async with asyncio.timeout(900):
            while not delivery.cancelled.is_set():
                event = await asyncio.to_thread(delivery.poll)
                if isinstance(event, bytes):
                    await socket.send_bytes(event)
                elif isinstance(event, Text):
                    await socket.send_json(
                        {"type": "assistant_text_delta", "text": event.value}
                    )
                elif isinstance(event, Ready):
                    await socket.send_json(
                        {
                            "type": "ready",
                            "setup_seconds": event.setup_seconds,
                            "input_sample_rate": 16000,
                            "output_sample_rate": 24000,
                        }
                    )
                elif isinstance(event, Metrics):
                    await socket.send_json({"type": "done", "metrics": asdict(event)})
                    break
                elif generation.done():
                    generation.result()
                    raise RuntimeError(
                        "Qwen generation ended without completion evidence"
                    )
    except WebSocketDisconnect, asyncio.CancelledError:
        pass
    except ValidationError as exc:
        errors = exc.errors(
            include_input=False, include_url=False, include_context=False
        )
        message = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in errors
        )
        with suppress(WebSocketDisconnect, RuntimeError):
            await socket.send_json({"type": "error", "message": message})
    except (ValueError, RuntimeError, TimeoutError) as exc:
        with suppress(WebSocketDisconnect, RuntimeError):
            await socket.send_json(
                {"type": "error", "message": str(exc) or "Qwen request timed out."}
            )
    except Exception:
        logger.exception("Qwen inference failed")
        with suppress(WebSocketDisconnect, RuntimeError):
            await socket.send_json(
                {"type": "error", "message": "Qwen inference failed."}
            )
    finally:
        delivery.cancelled.set()
        tasks = [task for task in (generation, stopping) if task is not None]
        for task in tasks:
            task.cancel()
        with CancelScope(shield=True):
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await worker.call(runtime.close)
            finally:
                worker.shutdown()
                with suppress(WebSocketDisconnect, RuntimeError):
                    await socket.close()
