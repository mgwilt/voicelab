"""Transport and cancellation tests without model weights or accelerators."""

import threading
import unittest
from collections.abc import Callable

import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from voicelab_core.synthesis import SynthesisMetrics
from voicelab_inference.sesame.torch.audio import Audio
from voicelab_models.sesame.model import SynthesisRequest
from voicelab_web.app import create_app
from voicelab_web.models.synthesis.service import Delivery, Failure, SynthesisService
from voicelab_web.schemas import ServerConfig


class FakeSynthesizer:
    def __init__(self, *, eos: bool = True, fail: bool = False) -> None:
        self.eos = eos
        self.fail = fail
        self.calls = 0

    def synthesize(
        self, request: SynthesisRequest, emit: Callable[[Audio], None]
    ) -> SynthesisMetrics:
        self.calls += 1
        if self.fail:
            raise ValueError("Deliberate test failure")
        emit(np.zeros(7680, dtype=np.float32))
        return SynthesisMetrics(0.16, 0.32, 0.5, 0.1, 0.01, 4, self.eos, ())


class ServiceTests(unittest.TestCase):
    def test_stream_completion_and_cutoff(self) -> None:
        for eos in (False, True):
            runtime = FakeSynthesizer(eos=eos)
            service = SynthesisService(
                SynthesisRequest, lambda runtime=runtime: runtime
            )
            with TestClient(create_app(ServerConfig(), sesame=service)) as client:
                self.assertEqual(client.get("/").status_code, 200)
                with client.websocket_connect(
                    "/api/sesame/session", headers={"origin": "http://testserver"}
                ) as ws:
                    ws.send_json({"text": "Hello"})
                    self.assertEqual(ws.receive_json()["sample_rate"], 24000)
                    self.assertEqual(len(ws.receive_bytes()), 15360)
                    result = ws.receive_json()
                    self.assertEqual(
                        result["type"], "complete" if eos else "incomplete"
                    )
                    self.assertEqual(result["metrics"]["output_rtf"], 0.5)
            self.assertFalse(service.lock.locked())

    def test_origin_validation_and_generation_failure(self) -> None:
        runtime = FakeSynthesizer(fail=True)
        service = SynthesisService(SynthesisRequest, lambda runtime=runtime: runtime)
        with TestClient(create_app(ServerConfig(), sesame=service)) as client:
            with (
                self.assertRaises(WebSocketDisconnect),
                client.websocket_connect(
                    "/api/sesame/session", headers={"origin": "https://other.invalid"}
                ),
            ):
                pass
            for request in ({"text": " "}, {"text": "Hello"}):
                with client.websocket_connect(
                    "/api/sesame/session", headers={"origin": "http://testserver"}
                ) as ws:
                    ws.send_json(request)
                    response = ws.receive_json()
                    if response["type"] == "ready":
                        response = ws.receive_json()
                    self.assertEqual(response["type"], "error")
        self.assertEqual(runtime.calls, 1)
        self.assertFalse(service.lock.locked())

    def test_busy_rejection_and_cancelled_backpressure(self) -> None:
        service = SynthesisService(SynthesisRequest, FakeSynthesizer)
        delivery = Delivery()
        service.lock.acquire()
        try:
            service.generate(SynthesisRequest(text="Hello"), delivery)
            self.assertIsInstance(delivery.poll(), Failure)
        finally:
            service.lock.release()
        for _ in range(8):
            delivery.publish(b"audio")
        cancelled = threading.Event()

        def blocked_producer() -> None:
            try:
                delivery.publish(b"overflow")
            except InterruptedError:
                cancelled.set()

        thread = threading.Thread(target=blocked_producer)
        thread.start()
        delivery.cancelled.set()
        thread.join(timeout=2)
        self.assertTrue(cancelled.is_set())
        self.assertFalse(thread.is_alive())
