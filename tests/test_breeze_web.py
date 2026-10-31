"""Breeze uses the same bounded synthesis transport with its own request schema."""

import unittest
from collections.abc import Callable

import numpy as np
from fastapi.testclient import TestClient

from voicelab_core.synthesis import SynthesisMetrics
from voicelab_models.breeze.synthesis import SynthesisRequest
from voicelab_web.app import create_app
from voicelab_web.models.synthesis.service import SynthesisService
from voicelab_web.schemas import ServerConfig
from voicelab_web.transport.pcm import Audio


class FakeBreeze:
    def __init__(self, complete: bool = True) -> None:
        self.complete = complete
        self.requests: list[SynthesisRequest] = []

    def synthesize(
        self, request: SynthesisRequest, emit: Callable[[Audio], None]
    ) -> SynthesisMetrics:
        self.requests.append(request)
        emit(np.zeros(7680, dtype=np.float32))
        return SynthesisMetrics(0.16, 0.32, 0.5, 0.1, 0.01, 4, self.complete, ())


class BreezeWebTests(unittest.TestCase):
    def test_instruction_request_stream_and_completion(self) -> None:
        runtime = FakeBreeze()
        service = SynthesisService(SynthesisRequest, lambda: runtime)
        with (
            TestClient(create_app(ServerConfig(), breeze=service)) as client,
            client.websocket_connect(
                "/api/breeze/session", headers={"origin": "http://testserver"}
            ) as socket,
        ):
            socket.send_json(
                {"text": "Hello", "instruction": "A calm voice", "cfg_scale": 3.5}
            )
            self.assertEqual(socket.receive_json()["sample_rate"], 24000)
            self.assertEqual(len(socket.receive_bytes()), 15360)
            result = socket.receive_json()
            self.assertEqual(result["type"], "complete")
            self.assertEqual(result["metrics"]["output_rtf"], 0.5)
        self.assertEqual(runtime.requests[0].instruction, "A calm voice")
        self.assertEqual(runtime.requests[0].cfg_scale, 3.5)
        self.assertFalse(service.lock.locked())

    def test_cap_is_incomplete_and_invalid_inputs_never_run(self) -> None:
        runtime = FakeBreeze(complete=False)
        service = SynthesisService(SynthesisRequest, lambda: runtime)
        with TestClient(create_app(ServerConfig(), breeze=service)) as client:
            for payload in ({"text": " "}, {"text": "Hello", "cfg_scale": 0.0}):
                with client.websocket_connect(
                    "/api/breeze/session", headers={"origin": "http://testserver"}
                ) as socket:
                    socket.send_json(payload)
                    self.assertEqual(socket.receive_json()["type"], "error")
            self.assertEqual(runtime.requests, [])
            with client.websocket_connect(
                "/api/breeze/session", headers={"origin": "http://testserver"}
            ) as socket:
                socket.send_json({"text": "Hello", "max_frames": 4})
                socket.receive_json()
                socket.receive_bytes()
                result = socket.receive_json()
                self.assertEqual(result["type"], "incomplete")
                self.assertFalse(result["metrics"]["eos"])
