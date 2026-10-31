"""Pipecat frame boundary with one serialized, cancellation-safe device owner."""

import asyncio
import contextlib
import threading
from functools import partial
from time import perf_counter
from typing import override

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    SpeechOutputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIProcessor

from voicelab_web.conversations.contracts import Audio, Backend, Notice
from voicelab_web.conversations.turns import TurnDetector
from voicelab_web.transport.worker import InferenceWorker


class ConversationProcessor(FrameProcessor):
    def __init__(self, backend: Backend, rtvi: RTVIProcessor) -> None:
        super().__init__()
        self.backend = backend
        self.rtvi = rtvi
        self.worker = InferenceWorker("conversation")
        self.detector = None if backend.duplex else TurnDetector()
        self.queue: asyncio.Queue[tuple[bytes, float, int]] = asyncio.Queue(
            maxsize=max(1, backend.input_rate * 4 // backend.frame_bytes)
            if backend.duplex
            else 1
        )
        self.buffer = bytearray()
        self.cancelled = threading.Event()
        self.stopping = threading.Event()
        self.ready = False
        self.loaded = asyncio.Event()
        self.job: asyncio.Task[None] | None = None
        self.close_task: asyncio.Task[None] | None = None
        self.turn = 0
        self.epoch = 0
        self.first_audio_seen = False
        self.input_started: float | None = None
        self.output_pending = False
        self.generating = False

    async def interrupt(self) -> None:
        if self.backend.duplex or not self.ready:
            return
        active = self.generating or self.output_pending or not self.queue.empty()
        self.epoch += 1
        self.cancelled.set()
        if self.generating or self.output_pending:
            self.backend.interrupt()
        self.output_pending = False
        while not self.queue.empty():
            self.queue.get_nowait()
        if active:
            await self.broadcast_interruption()
            await self.rtvi.send_server_message(
                Notice(type="interrupted", turn=self.turn).model_dump()
            )

    @override
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self.job = asyncio.create_task(self._run())
        elif isinstance(frame, (CancelFrame, EndFrame)):
            self.stopping.set()
            self.cancelled.set()
            await self.push_frame(frame, direction)
        elif isinstance(frame, InputAudioRawFrame):
            try:
                await self._input(frame)
            except ValueError as exc:
                self.stopping.set()
                self.cancelled.set()
                await self.rtvi.send_server_message(
                    Notice(type="error", text=str(exc)).model_dump()
                )
                await self.push_frame(ErrorFrame(str(exc), fatal=True))
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.output_pending = False
            if not self.generating and not self.cancelled.is_set():
                self.backend.played()
                await self.rtvi.send_server_message(
                    Notice(type="state", text="Listening").model_dump()
                )
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame) and self.backend.duplex:
            # Native duplex has its own interruption decisions and recurrent state.
            return
        else:
            await self.push_frame(frame, direction)

    async def _input(self, frame: InputAudioRawFrame) -> None:
        if not self.ready:
            return
        if frame.sample_rate != self.backend.input_rate or frame.num_channels != 1:
            raise ValueError("Conversation input has an incompatible PCM format")
        if self.backend.duplex:
            self.buffer.extend(frame.audio)
            size = self.backend.frame_bytes
            while len(self.buffer) >= size:
                self._enqueue(bytes(self.buffer[:size]))
                del self.buffer[:size]
        elif self.detector:
            began, turn = await self.detector.push(frame.audio)
            if began:
                await self.interrupt()
                await self.rtvi.send_server_message(
                    Notice(type="state", text="Listening").model_dump()
                )
            if turn is not None:
                self.turn += 1
                self._enqueue(turn)

    def _enqueue(self, pcm: bytes) -> None:
        if self.input_started is None:
            self.input_started = perf_counter()
        try:
            self.queue.put_nowait((pcm, perf_counter(), self.epoch))
        except asyncio.QueueFull as exc:
            raise ValueError("Inference cannot keep up with microphone audio") from exc

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()

        async def publish(event: Audio | Notice, epoch: int, turn: int) -> None:
            if epoch != self.epoch or self.stopping.is_set():
                return
            if isinstance(event, Audio):
                self.output_pending = True
                await self.push_frame(
                    (
                        SpeechOutputAudioRawFrame
                        if self.backend.duplex
                        else TTSAudioRawFrame
                    )(event.pcm, event.sample_rate, 1)
                )
            else:
                await self.rtvi.send_server_message(
                    event.model_copy(update={"turn": turn}).model_dump()
                )

        def emit(event: Audio | Notice, epoch: int, turn: int) -> None:
            if self.stopping.is_set() or epoch != self.epoch:
                raise InterruptedError("Conversation interrupted")
            future = asyncio.run_coroutine_threadsafe(publish(event, epoch, turn), loop)
            try:
                future.result(timeout=5)
            except BaseException:
                future.cancel()
                raise

        try:
            await self.worker.call(
                partial(
                    self.backend.load,
                    partial(emit, epoch=self.epoch, turn=0),
                    self.stopping.is_set,
                )
            )
            if self.stopping.is_set():
                return
            self.ready = True
            self.loaded.set()
            await self.rtvi.send_server_message(
                Notice(type="state", text="Listening").model_dump()
            )
            while not self.stopping.is_set():
                pcm, beginning, epoch = await self.queue.get()
                if epoch != self.epoch:
                    continue
                self.cancelled = threading.Event()
                cancelled = self.cancelled
                first = True
                turn = self.turn

                def deliver(
                    event: Audio | Notice,
                    cancelled: threading.Event = cancelled,
                    beginning: float = beginning,
                    epoch: int = epoch,
                    turn: int = turn,
                ) -> None:
                    nonlocal first
                    if cancelled.is_set():
                        raise InterruptedError("Conversation interrupted")
                    if (
                        isinstance(event, Audio)
                        and first
                        and (not self.backend.duplex or not self.first_audio_seen)
                    ):
                        first = False
                        self.first_audio_seen = True
                        emit(
                            Notice(
                                type="metrics",
                                stage="first_audio"
                                if self.backend.duplex
                                else "turn_to_first_audio",
                                seconds=perf_counter()
                                - (
                                    self.input_started
                                    if self.backend.duplex
                                    and self.input_started is not None
                                    else beginning
                                ),
                            ),
                            epoch,
                            turn,
                        )
                    emit(event, epoch, turn)

                self.generating = True
                try:
                    await self.worker.call(
                        partial(self.backend.process, pcm, deliver, cancelled.is_set)
                    )
                    if not self.backend.duplex and not cancelled.is_set():
                        await self.push_frame(TTSStoppedFrame())
                        if not self.output_pending:
                            self.backend.played()
                except InterruptedError:
                    self.backend.interrupt()
                    continue
                finally:
                    self.generating = False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.rtvi.send_server_message(
                Notice(type="error", text=str(exc)).model_dump()
            )
            await self.push_frame(ErrorFrame(str(exc), fatal=True))
        finally:
            self.ready = False

    async def close(self) -> None:
        if self.close_task is None:
            self.close_task = asyncio.create_task(self._close())
        await asyncio.shield(self.close_task)

    async def _close(self) -> None:
        self.stopping.set()
        self.cancelled.set()
        if self.job:
            self.job.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.job
        try:
            await self.worker.call(self.backend.close)
        finally:
            self.worker.shutdown()
            if self.detector:
                await self.detector.close()

    @override
    async def cleanup(self) -> None:
        await self.close()
        await super().cleanup()
