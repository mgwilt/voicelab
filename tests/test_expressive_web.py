"""Typed reference conditioning through the shared WebSocket boundary."""

import base64
import io
import unittest
import wave
from collections.abc import Callable

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from pydantic import BaseModel

from voicelab_core.synthesis import SynthesisMetrics
from voicelab_models.audio import ReferenceAudio
from voicelab_models.drama.model import PreviousTake, Reference
from voicelab_models.drama.model import Request as DramaRequest
from voicelab_models.kitten.model import Request as KittenRequest
from voicelab_models.maya.model import Request as MayaRequest
from voicelab_models.omnivoice.model import Request as OmniRequest
from voicelab_web.catalog_types import ModelId
from voicelab_web.models.synthesis.routes import SynthesisRoutes
from voicelab_web.models.synthesis.service import SynthesisService
from voicelab_web.transport.lease import SessionLease


def reference() -> ReferenceAudio:
    data = io.BytesIO()
    with wave.open(data, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(bytes(9600))
    return ReferenceAudio(wav_base64=base64.b64encode(data.getvalue()).decode("ascii"))


class Capture[Request: BaseModel]:
    def __init__(self) -> None:
        self.seen: Request | None = None

    def synthesize(
        self, request: Request, emit: Callable[[NDArray[np.float32]], None]
    ) -> SynthesisMetrics:
        self.seen = request
        emit(np.zeros(240, dtype=np.float32))
        return SynthesisMetrics(0.01, 0.01, 1, 0.01, None, 1, True, ())


async def release() -> None:
    pass


class ExpressiveWebTests(unittest.TestCase):
    def round_trip[Request: BaseModel](
        self, name: ModelId, request_type: type[Request], request: Request, rate: int
    ) -> None:
        backend = Capture[Request]()
        app = FastAPI()
        routes = SynthesisRoutes(app, SessionLease(), release)
        routes.add(name, SynthesisService(request_type, lambda: backend, rate))
        with (
            TestClient(app) as client,
            client.websocket_connect(
                f"/api/{name}/session", headers={"origin": "http://testserver"}
            ) as socket,
        ):
            socket.send_text(request.model_dump_json())
            self.assertEqual(
                socket.receive_json(), {"type": "ready", "sample_rate": rate}
            )
            self.assertEqual(len(socket.receive_bytes()), 480)
            self.assertEqual(socket.receive_json()["type"], "complete")
        self.assertEqual(backend.seen, request)

    def test_all_four_reference_requests_reach_typed_runtimes(self) -> None:
        audio = reference()
        self.round_trip(
            "drama",
            DramaRequest,
            DramaRequest(
                text="Hello",
                references=[Reference(audio=audio, text="Reference speech")],
                previous_takes=[PreviousTake(audio=audio, text="An earlier utterance")],
                top_k=20,
            ),
            44100,
        )
        self.round_trip(
            "maya",
            MayaRequest,
            MayaRequest(
                text="Hello",
                ref_audio=audio,
                ref_text="Reference speech",
                conditioning="icl",
                depth_temperature=0.5,
            ),
            24000,
        )
        self.round_trip(
            "omnivoice",
            OmniRequest,
            OmniRequest(
                text="Hello",
                ref_audio=audio,
                ref_text="Reference speech",
                duration=3.0,
                guidance_scale=1.5,
            ),
            24000,
        )
        self.round_trip(
            "kitten",
            KittenRequest,
            KittenRequest(
                text="Hello",
                ref_audio=audio,
                ref_text="Reference speech",
                use_reference_prompt=True,
                top_p=0.8,
            ),
            24000,
        )

    def test_invalid_reference_does_not_echo_recording_or_load_runtime(self) -> None:
        backend = Capture[KittenRequest]()
        app = FastAPI()
        routes = SynthesisRoutes(app, SessionLease(), release)
        routes.add("kitten", SynthesisService(KittenRequest, lambda: backend))
        audio = reference()
        request = {"text": "Hello", "ref_audio": audio.model_dump(), "voice": "Bella"}
        with (
            TestClient(app) as client,
            client.websocket_connect(
                "/api/kitten/session", headers={"origin": "http://testserver"}
            ) as socket,
        ):
            socket.send_json(request)
            result = socket.receive_json()
            self.assertEqual(result["type"], "error")
            self.assertNotIn(audio.wav_base64, str(result))
        self.assertIsNone(backend.seen)
