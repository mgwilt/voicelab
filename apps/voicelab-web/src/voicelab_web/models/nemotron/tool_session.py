"""Same-site offline Nemotron tool experiment; no external tool execution."""

import asyncio
import contextlib
import logging
import threading
import wave
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from anyio import CancelScope
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from voicelab_core.contracts import GenerationSettings
from voicelab_web.models.nemotron.engine import Engine
from voicelab_web.models.nemotron.tool_schema import ToolStart

logger = logging.getLogger(__name__)


async def serve_tool_session(socket: WebSocket, engine: Engine) -> None:
    await socket.accept()
    cancelled = threading.Event()
    generation: asyncio.Task[None] | None = None
    stopping: asyncio.Task[None] | None = None
    try:
        raw = await asyncio.wait_for(socket.receive_text(), 30)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("Tool experiment request exceeds 4 MiB.")
        request = ToolStart.model_validate_json(raw)
        await socket.send_json({"type": "loading"})

        async def wait_stop() -> None:
            try:
                await socket.receive()
            finally:
                cancelled.set()

        async def generate() -> None:
            with TemporaryDirectory(prefix="voicelab-nemotron-tools-") as temporary:
                source = Path(temporary) / "input.wav"
                destination = Path(temporary) / "response.wav"
                source.write_bytes(request.audio.wav_bytes())
                result = await engine.call(
                    partial(
                        engine.run_tool,
                        source,
                        destination,
                        GenerationSettings(
                            request.prompt, request.seed, request.trailing_silence
                        ),
                        request.settings(),
                        cancelled.is_set,
                    )
                )
                if cancelled.is_set():
                    return
                await socket.send_json(
                    {
                        "type": "ready",
                        "input_sample_rate": 16000,
                        "output_sample_rate": 22050,
                    }
                )
                await socket.send_json(
                    {"type": "function_delta", "text": result.call.serialized}
                )
                await socket.send_json(
                    {"type": "assistant_text_delta", "text": result.text}
                )
                with wave.open(str(destination), "rb") as audio:
                    while pcm := audio.readframes(16384):
                        await socket.send_bytes(pcm)
                elapsed = result.detection_seconds + result.continuation_seconds
                input_seconds = (
                    request.audio.duration_seconds + request.trailing_silence
                )
                await socket.send_json(
                    {
                        "type": "done",
                        "metrics": {
                            "rtf": elapsed / input_seconds,
                            "input_seconds": input_seconds,
                            "generation_seconds": elapsed,
                            "output_seconds": result.output_seconds,
                            "completed": result.completed,
                        },
                    }
                )

        generation = asyncio.create_task(generate())
        stopping = asyncio.create_task(wait_stop())
        async with asyncio.timeout(180):
            done, _ = await asyncio.wait(
                (generation, stopping), return_when=asyncio.FIRST_COMPLETED
            )
            if stopping in done:
                stopping.result()
                return
            generation.result()
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
    except TimeoutError:
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await socket.send_json(
                {"type": "error", "message": "Tool experiment timed out."}
            )
    except (ValueError, RuntimeError) as exc:
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await socket.send_json({"type": "error", "message": str(exc)})
    except Exception:
        logger.exception("Nemotron tool experiment failed")
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await socket.send_json(
                {"type": "error", "message": "Nemotron tool experiment failed."}
            )
    finally:
        cancelled.set()
        tasks = [task for task in (generation, stopping) if task is not None]
        for task in tasks:
            task.cancel()
        with CancelScope(shield=True):
            await asyncio.gather(*tasks, return_exceptions=True)
            await engine.call(engine.unload)
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                await socket.close()
