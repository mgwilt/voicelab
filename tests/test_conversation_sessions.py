"""Local WebSocket inference keeps ownership through cancellation and failure."""

import asyncio
import json
import threading
import unittest
from collections.abc import Callable
from unittest.mock import patch

from fastapi import WebSocket
from starlette.types import Message, Scope
from tests.test_conversation_adapters import request

from voicelab_inference.ming_audio.mlx.contracts import Response
from voicelab_inference.personaplex.mlx.runtime import Event, Metrics
from voicelab_models.ming_audio.model import Request as MingRequest
from voicelab_models.personaplex.model import FRAME_BYTES
from voicelab_models.personaplex.model import Request as PersonaRequest
from voicelab_web.models.ming_audio.session import serve_session as serve_ming
from voicelab_web.models.personaplex.session import serve_session as serve_persona


class Browser:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[Message] = asyncio.Queue()
        self.outgoing: list[Message] = []
        scope: Scope = {
            "type": "websocket",
            "path": "/",
            "scheme": "ws",
            "headers": [],
            "query_string": b"",
            "subprotocols": [],
        }
        self.socket = WebSocket(scope, self.incoming.get, self.send)
        self.incoming.put_nowait({"type": "websocket.connect"})

    async def send(self, message: Message) -> None:
        self.outgoing.append(message)

    def text(self, value: str) -> None:
        self.incoming.put_nowait({"type": "websocket.receive", "text": value})


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_ming_stop_waits_for_local_inference_cleanup(self) -> None:
        await self._cancel_ming(disconnect=False)

    async def test_ming_disconnect_waits_for_local_inference_cleanup(self) -> None:
        await self._cancel_ming(disconnect=True)

    async def _cancel_ming(self, *, disconnect: bool) -> None:
        entered = threading.Event()
        released = threading.Event()
        closed = threading.Event()
        owners: list[int] = []

        class Backend:
            def load(self) -> None:
                owners.append(threading.get_ident())

            def respond(
                self, value: MingRequest, should_cancel: Callable[[], bool]
            ) -> Response:
                owners.append(threading.get_ident())
                entered.set()
                try:
                    while not should_cancel():
                        if released.wait(0.01):
                            raise AssertionError("Unexpected release")
                    raise RuntimeError("Cancelled")
                finally:
                    released.set()

            def close(self) -> None:
                if not released.is_set():
                    raise AssertionError("Model closed during active inference")
                owners.append(threading.get_ident())
                closed.set()

        browser = Browser()
        browser.text(request().model_dump_json()[:-1] + ',"type":"start"}')
        with patch("voicelab_web.models.ming_audio.session.MingAudioMLX", Backend):
            task = asyncio.create_task(serve_ming(browser.socket))
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            if disconnect:
                browser.incoming.put_nowait(
                    {"type": "websocket.disconnect", "code": 1000}
                )
            else:
                browser.text('{"type":"stop"}')
            await asyncio.wait_for(task, 2)
        self.assertTrue(released.is_set())
        self.assertTrue(closed.is_set())
        self.assertEqual(len(set(owners)), 1)
        self.assertEqual(browser.outgoing[-1]["type"], "websocket.close")

    async def test_validation_errors_do_not_echo_reference_recordings(self) -> None:
        for serve in (serve_ming, serve_persona):
            browser = Browser()
            browser.text(
                json.dumps(
                    {
                        "type": "start",
                        "reference_audio": {
                            "wav_base64": "PRIVATE_REFERENCE_PAYLOAD_NOT_BASE64"
                        },
                    }
                )
            )
            await asyncio.wait_for(serve(browser.socket), 2)
            outgoing = json.dumps(browser.outgoing)
            self.assertIn("error", outgoing)
            self.assertNotIn("PRIVATE_REFERENCE_PAYLOAD", outgoing)

    async def test_personaplex_failure_closes_local_model(self) -> None:
        released = threading.Event()

        class Backend:
            def start(self, value: PersonaRequest) -> None:
                pass

            def push(self, pcm: bytes) -> tuple[Event, ...]:
                raise TimeoutError("Model stalled")

            def metrics(self) -> Metrics:
                return Metrics(0.0, 0.0, 0.0, None, None)

            def close(self) -> None:
                released.set()

        browser = Browser()
        browser.text('{"type":"start"}')
        browser.incoming.put_nowait(
            {"type": "websocket.receive", "bytes": bytes(FRAME_BYTES)}
        )
        with patch("voicelab_web.models.personaplex.session.PersonaPlexMLX", Backend):
            await asyncio.wait_for(serve_persona(browser.socket), 2)
        self.assertTrue(released.is_set())
        self.assertTrue(any("Model stalled" in str(item) for item in browser.outgoing))
        self.assertEqual(browser.outgoing[-1]["type"], "websocket.close")


if __name__ == "__main__":
    unittest.main()
