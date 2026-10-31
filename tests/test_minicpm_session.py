"""MiniCPM website request, delivery and cancellation ownership."""

import threading
import unittest
from collections.abc import Callable
from typing import override
from unittest.mock import patch

from fastapi import FastAPI, WebSocketDisconnect
from fastapi.testclient import TestClient

from voicelab_inference.minicpm.native.protocol import Event
from voicelab_inference.minicpm.native.runtime import MiniCPMNative, never_cancel
from voicelab_models.minicpm.model import Request
from voicelab_web.models.minicpm.session import serve_session


class FakeRuntime(MiniCPMNative):
    def __init__(self) -> None:
        super().__init__()
        self.closed = threading.Event()

    @override
    def start(
        self, request: Request, cancelled: Callable[[], bool] = never_cancel
    ) -> None:
        self._request = request

    @override
    def turn(
        self,
        emit: Callable[[Event], None],
        cancelled: Callable[[], bool] = never_cancel,
        *,
        request: Request | None = None,
    ) -> None:
        emit(Event("text", text="Hello"))
        emit(Event("audio", pcm=bytes(480)))
        self._output = 0.01
        self._completed = True

    @override
    def finish(
        self,
        emit: Callable[[Event], None],
        cancelled: Callable[[], bool] = never_cancel,
    ) -> None:
        pass

    @override
    def close(self) -> None:
        self.closed.set()


def app() -> FastAPI:
    application = FastAPI()
    application.add_api_websocket_route("/minicpm", serve_session)
    return application


class MiniCPMSessionTests(unittest.TestCase):
    def test_pcm_and_completion(self) -> None:
        runtime = FakeRuntime()
        with (
            patch(
                "voicelab_web.models.minicpm.session.MiniCPMNative",
                return_value=runtime,
            ),
            TestClient(app()) as client,
            client.websocket_connect("/minicpm") as socket,
        ):
            socket.send_json({"type": "start", "mode": "turn_based", "text": "Hello"})
            self.assertEqual(socket.receive_json()["type"], "loading")
            self.assertEqual(socket.receive_json()["type"], "ready")
            self.assertEqual(socket.receive_json()["text"], "Hello")
            self.assertEqual(len(socket.receive_bytes()), 480)
            self.assertTrue(socket.receive_json()["metrics"]["completed"])
        self.assertTrue(runtime.closed.is_set())

    def test_invalid_input_before_loading(self) -> None:
        runtime = FakeRuntime()
        with (
            patch(
                "voicelab_web.models.minicpm.session.MiniCPMNative",
                return_value=runtime,
            ),
            patch.object(runtime, "start") as start,
            TestClient(app()) as client,
            client.websocket_connect("/minicpm") as socket,
        ):
            socket.send_json({"type": "start", "mode": "turn_based"})
            self.assertEqual(socket.receive_json()["type"], "error")
            start.assert_not_called()
        self.assertTrue(runtime.closed.is_set())

    def test_disconnect_during_setup_cancels_before_cleanup(self) -> None:
        entered = threading.Event()
        observed = threading.Event()

        class LoadingRuntime(FakeRuntime):
            @override
            def start(
                self, request: Request, cancelled: Callable[[], bool] = never_cancel
            ) -> None:
                entered.set()
                for _ in range(100):
                    if cancelled():
                        observed.set()
                        raise InterruptedError("cancelled")
                    threading.Event().wait(0.01)
                raise TimeoutError("test did not receive cancellation")

        runtime = LoadingRuntime()
        with (
            patch(
                "voicelab_web.models.minicpm.session.MiniCPMNative",
                return_value=runtime,
            ),
            TestClient(app()) as client,
        ):
            with client.websocket_connect("/minicpm") as socket:
                socket.send_json({"type": "start"})
                self.assertEqual(socket.receive_json()["type"], "loading")
                self.assertTrue(entered.wait(2))
                socket.send_json({"type": "stop"})
                with self.assertRaises(WebSocketDisconnect):
                    socket.receive_json()
            self.assertTrue(observed.wait(2))
        self.assertTrue(runtime.closed.is_set())
