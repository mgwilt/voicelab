"""Reject invalid tool uploads before load and retain ownership through cancellation."""

import asyncio
import base64
import json
import threading
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, override
from unittest.mock import patch

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from tests.test_conversation_adapters import wav_bytes

from voicelab_core.contracts import GenerationSettings
from voicelab_models.nemotron.tools import ToolSettings
from voicelab_web.models.nemotron.engine import Engine
from voicelab_web.models.nemotron.tool_session import serve_tool_session

if TYPE_CHECKING:
    from voicelab_inference.nemotron.mlx.tool_result import ToolResult


def request() -> dict[str, object]:
    return {
        "type": "start",
        "audio": {"wav_base64": base64.b64encode(wav_bytes(frames=16000)).decode()},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Read a value",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "api_response": {"tool_name": "lookup", "response": {"value": 73}},
    }


class ControlledEngine(Engine):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.finished = threading.Event()
        self.unloaded = threading.Event()

    @override
    def run_tool(
        self,
        source: Path,
        destination: Path,
        settings: GenerationSettings,
        tools: ToolSettings,
        should_cancel: Callable[[], bool],
    ) -> ToolResult:
        self.started.set()
        while not should_cancel():
            self.finished.wait(0.005)
        self.finished.set()
        raise InterruptedError("Cancelled")

    @override
    def unload(self) -> None:
        if self.started.is_set() and not self.finished.is_set():
            raise RuntimeError("Ownership released before worker finished")
        self.unloaded.set()


def application(engine: Engine) -> FastAPI:
    app = FastAPI()

    @app.websocket("/tools")
    async def tools(socket: WebSocket) -> None:
        await serve_tool_session(socket, engine)

    return app


class NemotronToolSessionTests(unittest.TestCase):
    def test_invalid_response_rejected_before_model_load_without_payload_echo(
        self,
    ) -> None:
        engine = ControlledEngine()
        value = request()
        value["api_response"] = {"tool_name": "other", "response": "private-payload"}
        try:
            with (
                TestClient(application(engine)) as client,
                client.websocket_connect("/tools") as socket,
            ):
                socket.send_json(value)
                result = socket.receive_json()
                self.assertEqual(result["type"], "error")
                self.assertNotIn("private-payload", json.dumps(result))
            self.assertFalse(engine.started.is_set())
        finally:
            engine.shutdown()

    def test_cancel_waits_for_worker_before_unload(self) -> None:
        engine = ControlledEngine()
        try:
            with (
                TestClient(application(engine)) as client,
                client.websocket_connect("/tools") as socket,
            ):
                socket.send_json(request())
                self.assertEqual(socket.receive_json()["type"], "loading")
                self.assertTrue(engine.started.wait(2))
                socket.send_json({"type": "stop"})
                self.assertEqual(socket.receive()["type"], "websocket.close")
            self.assertTrue(engine.finished.is_set())
            self.assertTrue(engine.unloaded.is_set())
        finally:
            engine.shutdown()

    def test_timeout_finishes_worker_before_cleanup(self) -> None:
        engine = ControlledEngine()
        real_timeout = asyncio.timeout
        try:
            with (
                patch(
                    "voicelab_web.models.nemotron.tool_session.asyncio.timeout",
                    side_effect=lambda _: real_timeout(0.02),
                ),
                TestClient(application(engine)) as client,
                client.websocket_connect("/tools") as socket,
            ):
                socket.send_json(request())
                self.assertEqual(socket.receive_json()["type"], "loading")
                self.assertEqual(socket.receive_json()["type"], "error")
                self.assertEqual(socket.receive()["type"], "websocket.close")
            self.assertTrue(engine.finished.is_set())
            self.assertTrue(engine.unloaded.is_set())
        finally:
            engine.shutdown()

    def test_oversized_request_rejected_before_model_load(self) -> None:
        engine = ControlledEngine()
        try:
            with (
                TestClient(application(engine)) as client,
                client.websocket_connect("/tools") as socket,
            ):
                socket.send_text(" " * (4 * 1024 * 1024 + 1))
                self.assertEqual(socket.receive_json()["type"], "error")
            self.assertFalse(engine.started.is_set())
        finally:
            engine.shutdown()
