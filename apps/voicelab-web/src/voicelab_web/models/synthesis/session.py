"""One WebSocket synthesis protocol for model-specific typed requests."""

import asyncio
from contextlib import suppress
from dataclasses import asdict

from anyio import CancelScope
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from voicelab_core.synthesis import SynthesisMetrics
from voicelab_web.models.synthesis.service import Delivery, Failure, SynthesisService


async def serve_session[Request: BaseModel](
    socket: WebSocket, service: SynthesisService[Request]
) -> None:
    expected = f"{'https' if socket.url.scheme == 'wss' else 'http'}://{socket.headers.get('host')}"
    if socket.headers.get("origin") != expected:
        await socket.close(code=1008)
        return
    await socket.accept()
    delivery = Delivery()
    worker: asyncio.Task[None] | None = None
    listener: asyncio.Task[None] | None = None
    try:
        raw = await asyncio.wait_for(socket.receive_text(), timeout=15)
        if len(raw) > 17 * 1024 * 1024:
            raise ValueError("Request too large")
        request = service.request_type.model_validate_json(raw)
        await socket.send_json({"type": "ready", "sample_rate": service.sample_rate})
        worker = asyncio.create_task(
            asyncio.to_thread(service.generate, request, delivery)
        )

        async def watch_disconnect() -> None:
            try:
                await socket.receive()
            finally:
                delivery.cancelled.set()

        listener = asyncio.create_task(watch_disconnect())
        async with asyncio.timeout(180):
            while not delivery.cancelled.is_set():
                event = await asyncio.to_thread(delivery.poll)
                if isinstance(event, bytes):
                    await socket.send_bytes(event)
                elif isinstance(event, SynthesisMetrics):
                    await socket.send_json(
                        {
                            "type": event.completion_state,
                            "metrics": asdict(event),
                        }
                    )
                    break
                elif isinstance(event, Failure):
                    await socket.send_json({"type": "error", "message": event.message})
                    break
                elif worker.done():
                    break
    except TimeoutError:
        with suppress(WebSocketDisconnect, RuntimeError):
            await socket.send_json(
                {"type": "error", "message": "Synthesis session timed out."}
            )
    except ValidationError as exc:
        errors = exc.errors(
            include_input=False, include_url=False, include_context=False
        )
        message = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in errors
        )
        with suppress(WebSocketDisconnect):
            await socket.send_json({"type": "error", "message": message})
    except ValueError as exc:
        with suppress(WebSocketDisconnect):
            await socket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        pass
    finally:
        with CancelScope(shield=True):
            delivery.cancelled.set()
            if listener is not None:
                listener.cancel()
                await asyncio.gather(listener, return_exceptions=True)
            if worker is not None:
                # Do not release ownership while a device kernel is still running.
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    await worker
                    raise
            with suppress(RuntimeError, WebSocketDisconnect):
                await socket.close()
