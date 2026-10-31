"""Exercise transport validation and session ownership without model weights."""

import unittest
from typing import override

from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from voicelab_models.nemotron.streaming import FRAME_BYTES, StreamEvent
from voicelab_web.app import create_app
from voicelab_web.models.nemotron.engine import Engine
from voicelab_web.models.nemotron.metrics import FrameMetrics
from voicelab_web.models.nemotron.schemas import StartMessage
from voicelab_web.schemas import ServerConfig


class FakeEngine(Engine):
    def __init__(self) -> None:
        super().__init__()
        self.closed = 0

    @override
    def start(self, prompt: str, seed: int) -> float:
        return 0.01

    @override
    def push(self, pcm: bytes) -> tuple[tuple[StreamEvent, ...], float]:
        return (StreamEvent("audio", pcm=b"\0\0" * 1764),), 0.04

    @override
    def close_session(self) -> None:
        self.closed += 1


class WebTests(unittest.TestCase):
    def test_rtf_uses_input_timeline(self) -> None:
        metrics = FrameMetrics()
        self.assertIsNone(metrics.summary()["rtf"])
        metrics.add(0.04)
        metrics.add(0.08)
        rtf = metrics.summary()["rtf"]
        assert isinstance(rtf, float)
        self.assertAlmostEqual(rtf, 0.75)
        self.assertEqual(metrics.summary()["input_seconds"], 0.16)

    def test_strict_start_validation(self) -> None:
        for value in (
            '{"type":"start","seed":"1"}',
            '{"type":"start","unknown":1}',
            '{"type":"start","prompt":"你好"}',
        ):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                StartMessage.model_validate_json(value)

    def test_duplex_audio_metrics_and_cleanup(self) -> None:
        engine = FakeEngine()
        with TestClient(create_app(ServerConfig(), engine)) as client:
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.get("/.local/server.toml").status_code, 404)
            with client.websocket_connect("/api/nemotron/session") as socket:
                socket.send_json({"type": "start"})
                self.assertEqual(socket.receive_json()["type"], "loading")
                self.assertEqual(socket.receive_json()["type"], "ready")
                socket.send_bytes(b"\0" * FRAME_BYTES)
                self.assertEqual(len(socket.receive_bytes()), 3528)
                self.assertAlmostEqual(socket.receive_json()["rtf"], 0.5)
                socket.send_json({"type": "stop"})
                self.assertEqual(socket.receive_json()["type"], "done")
            self.assertFalse(engine.busy)
            self.assertGreaterEqual(engine.closed, 1)

    def test_second_session_is_rejected(self) -> None:
        with (
            TestClient(create_app(ServerConfig(), FakeEngine())) as client,
            client.websocket_connect("/api/nemotron/session") as first,
        ):
            first.send_json({"type": "start"})
            first.receive_json()
            first.receive_json()
            with client.websocket_connect("/api/nemotron/session") as second:
                self.assertEqual(second.receive_json()["type"], "error")

    def test_invalid_audio_is_rejected(self) -> None:
        with (
            TestClient(create_app(ServerConfig(), FakeEngine())) as client,
            client.websocket_connect("/api/nemotron/session") as socket,
        ):
            socket.send_json({"type": "start"})
            socket.receive_json()
            socket.receive_json()
            socket.send_bytes(b"wrong size")
            self.assertEqual(socket.receive_json()["type"], "error")

    def test_foreign_browser_origin_is_rejected(self) -> None:
        with (
            TestClient(create_app(ServerConfig(), FakeEngine())) as client,
            self.assertRaises(WebSocketDisconnect),
            client.websocket_connect(
                "/api/nemotron/session", headers={"origin": "https://example.org"}
            ),
        ):
            self.fail("Foreign origin accepted")

    def test_one_site_catalog_and_compiled_assets(self) -> None:
        with TestClient(create_app(ServerConfig(), FakeEngine())) as client:
            catalog = client.get("/api/models")
            self.assertEqual(catalog.status_code, 200)
            rows = catalog.json()
            self.assertEqual(
                [row["id"] for row in rows], ["nemotron", "sesame", "breeze"]
            )
            self.assertEqual(
                [row["output_sample_rate"] for row in rows], [22050, 24000, 24000]
            )
            self.assertEqual(client.get("/static/dist/main.js").status_code, 200)
            self.assertEqual(
                client.get("/static/dist/capture-worklet.js").status_code, 200
            )
            self.assertEqual(client.get("/static/app.js").status_code, 404)
            self.assertEqual(client.get("/health").json()["active_model"], None)

    def test_one_lease_across_different_models(self) -> None:
        with (
            TestClient(create_app(ServerConfig(), FakeEngine())) as client,
            client.websocket_connect("/api/nemotron/session") as first,
        ):
            first.send_json({"type": "start"})
            first.receive_json()
            first.receive_json()
            self.assertEqual(client.get("/health").json()["active_model"], "nemotron")
            for route in ("/api/sesame/session", "/api/breeze/session"):
                with client.websocket_connect(
                    route, headers={"origin": "http://testserver"}
                ) as second:
                    message = second.receive_json()
                    self.assertEqual(message["type"], "error")
                    self.assertIn("Another model session", message["message"])
            first.send_json({"type": "stop"})
            self.assertEqual(first.receive_json()["type"], "done")
