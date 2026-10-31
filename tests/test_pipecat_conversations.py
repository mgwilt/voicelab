"""Conversation boundaries without checkpoints, microphones, or a network."""

import asyncio
import threading
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from voicelab_web.conversations.history import History
from voicelab_web.conversations.requests import REQUEST
from voicelab_web.conversations.routes import Conversations, install
from voicelab_web.transport.lease import SessionLease
from voicelab_web.transport.worker import InferenceWorker


async def release() -> None:
    pass


class HistoryTests(unittest.TestCase):
    def test_unplayed_and_interrupted_answers_are_not_context(self) -> None:
        history = History[str](limit=4)
        history.stage("question", "answer")
        self.assertEqual(history.snapshot(), ())
        history.played()
        history.stage("new question", "unheard answer")
        history.interrupt()
        history.played()
        self.assertEqual(history.snapshot(), ("question", "answer"))

    def test_context_evicts_whole_exchanges(self) -> None:
        history = History[str](limit=4)
        for n in range(3):
            history.stage(f"q{n}", f"a{n}")
            history.played()
        self.assertEqual(history.snapshot(), ("q1", "a1", "q2", "a2"))


class RequestTests(unittest.TestCase):
    def test_all_models_have_valid_defaults_and_distinct_paradigms(self) -> None:
        expected = {
            "nemotron": "duplex",
            "personaplex": "duplex",
            "minicpm": "duplex",
            "qwen_omni": "turn",
            "ming_audio": "turn",
            **dict.fromkeys(
                (
                    "sesame",
                    "breeze",
                    "phonon",
                    "kokoro",
                    "audio8",
                    "ming_tts",
                    "drama",
                    "maya",
                    "omnivoice",
                    "kitten",
                ),
                "cascade",
            ),
        }
        for model, mode in expected.items():
            with self.subTest(model=model):
                request = REQUEST.validate_python({"model": model})
                self.assertEqual(request.mode, mode)
                with self.assertRaises(ValidationError):
                    REQUEST.validate_python({"model": model, "unexpected": True})
        with self.assertRaises(ValidationError):
            REQUEST.validate_python({"model": "nemotron", "mode": "cascade"})
        with self.assertRaises(ValidationError):
            REQUEST.validate_python(
                {"model": "kokoro", "settings": {"text": "x", "speed": float("nan")}}
            )

    def test_lease_cannot_be_released_by_an_old_session(self) -> None:
        lease = SessionLease()
        first = lease.acquire("kokoro")
        with self.assertRaises(RuntimeError):
            lease.acquire("nemotron")
        lease.release(first)
        second = lease.acquire("nemotron")
        lease.release(first)
        self.assertEqual(lease.active, "nemotron")
        lease.release(second)
        self.assertIsNone(lease.active)

    def test_signaling_validates_origin_before_allocating(self) -> None:
        lease = SessionLease()

        def no_backend(request: object) -> None:
            raise AssertionError("Signaling validation must not construct a model")

        # Factory is never invoked in these reservation-only tests.
        from voicelab_web.conversations.contracts import Backend

        def factory(request: object) -> Backend:
            no_backend(request)
            raise AssertionError("unreachable")

        manager = Conversations(lease, release, factory)
        app = FastAPI()
        install(app, manager)
        with (
            TestClient(app) as client,
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
        ):
            self.assertEqual(
                client.post("/api/conversations", json={"model": "kokoro"}).status_code,
                403,
            )
            headers = {"origin": "http://testserver"}
            invalid = client.post(
                "/api/conversations", headers=headers, json={"model": "unknown"}
            )
            self.assertEqual(invalid.status_code, 422)
            self.assertIsNone(lease.active)
            response = client.post(
                "/api/conversations", headers=headers, json={"model": "kokoro"}
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(lease.active, "kokoro")
            busy = client.post(
                "/api/conversations", headers=headers, json={"model": "nemotron"}
            )
            self.assertEqual(busy.status_code, 409)
            created = response.json()
            endpoint = f"/api/conversations/{created['session_id']}"
            self.assertEqual(client.delete(endpoint, headers=headers).status_code, 204)
            self.assertIsNone(lease.active)
            self.assertEqual(client.delete(endpoint, headers=headers).status_code, 204)


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_waits_for_uninterruptible_device_work(self) -> None:
        started, finish = threading.Event(), threading.Event()
        worker = InferenceWorker("test-conversation")

        def work() -> None:
            started.set()
            finish.wait(2)

        task = asyncio.create_task(worker.call(work))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        finish.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        worker.shutdown()

    async def test_unoffered_reservation_expires(self) -> None:
        from voicelab_web.conversations.contracts import Backend

        def unused(request: object) -> Backend:
            raise AssertionError("No model should load before signaling")

        lease = SessionLease()
        manager = Conversations(lease, release, unused)
        loop = asyncio.get_running_loop()
        original = loop.call_later
        with patch.object(
            loop,
            "call_later",
            side_effect=lambda delay, callback: original(0, callback),
        ):
            manager.create(REQUEST.validate_python({"model": "kokoro"}))
        await asyncio.sleep(0.01)
        self.assertIsNone(lease.active)
        self.assertEqual(manager.sessions, {})


class FrameBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplex_reassembles_transport_blocks_without_turn_gating(
        self,
    ) -> None:
        from pipecat.frames.frames import InputAudioRawFrame
        from pipecat.processors.frameworks.rtvi import RTVIProcessor

        from voicelab_web.conversations.contracts import Cancelled, Emit
        from voicelab_web.conversations.processor import ConversationProcessor

        class Duplex:
            input_rate = 16000
            frame_bytes = 2560
            duplex = True

            def load(self, emit: Emit, cancelled: Cancelled) -> None:
                pass

            def process(self, pcm: bytes, emit: Emit, cancelled: Cancelled) -> None:
                pass

            def interrupt(self) -> None:
                raise AssertionError("Native duplex must not use turn interruption")

            def played(self) -> None:
                pass

            def close(self) -> None:
                pass

        processor = ConversationProcessor(Duplex(), RTVIProcessor())
        try:
            processor.ready = True
            self.assertIsNone(processor.detector)
            await processor._input(InputAudioRawFrame(bytes(640), 16000, 1))
            self.assertTrue(processor.queue.empty())
            await processor._input(InputAudioRawFrame(bytes(1920), 16000, 1))
            pcm, _, epoch = processor.queue.get_nowait()
            self.assertEqual(pcm, bytes(2560))
            await processor.interrupt()
            self.assertEqual(processor.epoch, epoch)
            with self.assertRaises(ValueError):
                await processor._input(InputAudioRawFrame(bytes(640), 24000, 1))
            for _ in range(processor.queue.maxsize):
                await processor._input(InputAudioRawFrame(bytes(2560), 16000, 1))
            with self.assertRaisesRegex(ValueError, "cannot keep up"):
                await processor._input(InputAudioRawFrame(bytes(2560), 16000, 1))
        finally:
            await processor.close()

    async def test_concurrent_close_waits_for_the_same_device_cleanup(self) -> None:
        from pipecat.processors.frameworks.rtvi import RTVIProcessor

        from voicelab_web.conversations.contracts import Cancelled, Emit
        from voicelab_web.conversations.processor import ConversationProcessor

        entered, release_device = threading.Event(), threading.Event()

        class Backend:
            input_rate = 16000
            frame_bytes = 2560
            duplex = True
            closes = 0

            def load(self, emit: Emit, cancelled: Cancelled) -> None:
                pass

            def process(self, pcm: bytes, emit: Emit, cancelled: Cancelled) -> None:
                pass

            def interrupt(self) -> None:
                pass

            def played(self) -> None:
                pass

            def close(self) -> None:
                self.closes += 1
                entered.set()
                release_device.wait(2)

        backend = Backend()
        processor = ConversationProcessor(backend, RTVIProcessor())
        first = asyncio.create_task(processor.close())
        await asyncio.to_thread(entered.wait, 1)
        first.cancel()
        second = asyncio.create_task(processor.close())
        await asyncio.sleep(0)
        self.assertFalse(second.done())
        release_device.set()
        with self.assertRaises(asyncio.CancelledError):
            await first
        await second
        self.assertEqual(backend.closes, 1)


class TurnDetectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_hysteresis_does_not_end_a_turn_on_the_first_quiet_frame(
        self,
    ) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
        from pipecat.audio.vad.vad_analyzer import VADState

        from voicelab_web.conversations.turns import TurnDetector

        vad = MagicMock()
        vad.analyze_audio = AsyncMock(
            side_effect=[
                VADState.STARTING,
                VADState.SPEAKING,
                VADState.STOPPING,
                VADState.QUIET,
            ]
        )
        vad.cleanup = AsyncMock()
        smart = MagicMock()
        smart.append_audio.return_value = EndOfTurnState.INCOMPLETE
        smart.analyze_end_of_turn = AsyncMock(
            return_value=(EndOfTurnState.COMPLETE, None)
        )
        smart.cleanup = AsyncMock()
        with (
            patch(
                "voicelab_web.conversations.turns.SileroVADAnalyzer", return_value=vad
            ),
            patch(
                "voicelab_web.conversations.turns.LocalSmartTurnAnalyzerV3",
                return_value=smart,
            ),
        ):
            detector = TurnDetector()
        try:
            for index in range(3):
                began, turn = await detector.push(bytes(640))
                self.assertEqual(began, index == 1)
                self.assertIsNone(turn)
                smart.analyze_end_of_turn.assert_not_awaited()
            began, turn = await detector.push(bytes(640))
            self.assertFalse(began)
            self.assertEqual(turn, bytes(2560))
            smart.analyze_end_of_turn.assert_awaited_once()
        finally:
            await detector.close()
