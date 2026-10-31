"""Qwen WebSocket validation, PCM delivery and loader ownership."""

import threading
import unittest
from typing import override
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from tests.test_qwen_omni import FakeBackend

from voicelab_inference.qwen_omni.mlx.runtime import QwenOmniMLX
from voicelab_web.models.qwen_omni.session import serve_session


class QwenSessionTests(unittest.TestCase):
    def test_websocket_reports_audio_and_verified_completion(self) -> None:
        app = FastAPI()
        app.add_api_websocket_route("/qwen", serve_session)
        backend = FakeBackend()
        with (
            patch(
                "voicelab_web.models.qwen_omni.session.QwenOmniMLX",
                return_value=QwenOmniMLX(factory=lambda: backend),
            ),
            TestClient(app) as client,
            client.websocket_connect("/qwen") as socket,
        ):
            socket.send_json({"type": "start", "prompt": "Hello"})
            self.assertEqual(socket.receive_json()["type"], "loading")
            self.assertEqual(socket.receive_json()["type"], "ready")
            self.assertEqual(socket.receive_json()["text"], "Hello")
            self.assertEqual(
                len(socket.receive_bytes()) + len(socket.receive_bytes()), 48000
            )
            done = socket.receive_json()
            self.assertEqual(done["type"], "done")
            self.assertTrue(done["metrics"]["completed"])
            with self.assertRaises(WebSocketDisconnect):
                socket.receive_json()
        self.assertTrue(backend.closed)

    def test_stop_during_load_waits_for_worker_then_closes(self) -> None:
        entered, release, closed = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        class LoadingRuntime(QwenOmniMLX):
            @override
            def load(self) -> None:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test loader was not released")

            @override
            def close(self) -> None:
                closed.set()

        app = FastAPI()
        app.add_api_websocket_route("/qwen", serve_session)
        with (
            patch(
                "voicelab_web.models.qwen_omni.session.QwenOmniMLX",
                return_value=LoadingRuntime(),
            ),
            TestClient(app) as client,
            client.websocket_connect("/qwen") as socket,
        ):
            socket.send_json({"type": "start"})
            self.assertEqual(socket.receive_json()["type"], "loading")
            self.assertTrue(entered.wait(2))
            socket.send_json({"type": "stop"})
            try:
                self.assertFalse(closed.wait(0.2))
            finally:
                release.set()
            with self.assertRaises(WebSocketDisconnect):
                socket.receive_json()
        self.assertTrue(closed.is_set())

    def test_invalid_request_does_not_load(self) -> None:
        app = FastAPI()
        app.add_api_websocket_route("/qwen", serve_session)
        backend = FakeBackend()
        runtime = QwenOmniMLX(factory=lambda: backend)
        with (
            patch(
                "voicelab_web.models.qwen_omni.session.QwenOmniMLX",
                return_value=runtime,
            ),
            patch.object(runtime, "load") as load,
            TestClient(app) as client,
            client.websocket_connect("/qwen") as socket,
        ):
            socket.send_json({"type": "start", "speaker": "unknown"})
            self.assertEqual(socket.receive_json()["type"], "error")
            load.assert_not_called()
