"""MiniCPM speech modes through one owned native inference session."""

import base64
import json
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from websockets.sync.client import ClientConnection, connect

from voicelab_inference.audio import reference_pcm
from voicelab_inference.minicpm.native.prepare import prepare
from voicelab_inference.minicpm.native.process import NativeProcess
from voicelab_inference.minicpm.native.protocol import (
    Event,
    NativeMessage,
    decode_audio,
    encode_audio,
)
from voicelab_models.audio import ReferenceAudio
from voicelab_models.minicpm.model import (
    FRAME_BYTES,
    INPUT_RATE,
    MAX_SECONDS,
    OUTPUT_RATE,
    Request,
)


def never_cancel() -> bool:
    return False


def reference_encoded(reference: ReferenceAudio) -> str:
    return base64.b64encode(
        reference_pcm(reference, INPUT_RATE).astype("<f4").tobytes()
    ).decode("ascii")


@dataclass(frozen=True)
class Metrics:
    input_seconds: float
    output_seconds: float
    generation_seconds: float
    rtf: float | None
    first_pcm_seconds: float | None
    completed: bool | None


class MiniCPMNative:
    implementation = "minicpm/llama.cpp-omni"
    device = "metal"
    precision = "Q4_K_M language model; F16 speech modules"

    def __init__(self, *, assets: tuple[Path, Path] | None = None) -> None:
        self._assets = assets
        self._process = NativeProcess()
        self._socket: ClientConnection | None = None
        self._request: Request | None = None
        self._input = 0.0
        self._output = 0.0
        self._elapsed = 0.0
        self._first: float | None = None
        self._completed: bool | None = None
        self.setup_seconds = 0.0

    def start(
        self, request: Request, cancelled: Callable[[], bool] = never_cancel
    ) -> None:
        if self._socket is not None:
            raise RuntimeError("MiniCPM session is already active.")
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError(
                "This MiniCPM native adapter currently supports Apple Silicon Metal."
            )
        start = time.perf_counter()
        self._input = self._output = self._elapsed = 0.0
        self._first = self._completed = None
        self._request = request
        try:
            binary, checkpoint = self._assets or prepare(cancelled)
            self._process.start(binary, checkpoint, request, cancelled)
            self._socket = connect(
                f"ws://127.0.0.1:{self._process.port}/backend",
                additional_headers=self._process.headers,
                open_timeout=15,
                max_size=16 * 1024 * 1024,
                close_timeout=1,
            )
            voice = (
                reference_encoded(request.reference_audio)
                if request.reference_audio
                else None
            )
            self._socket.send(
                json.dumps(
                    {
                        "type": "session.init",
                        "payload": {
                            "mode": request.mode,
                            "system_prompt": request.prompt,
                            "voice": {"ref_audio": voice},
                            "use_tts": True,
                            "config": {
                                "length_penalty": request.length_penalty,
                                "listen_prob_scale": request.listen_prob_scale,
                                "max_new_speak_tokens_per_chunk": (
                                    request.max_speak_tokens
                                ),
                                "tts_temperature": request.tts_temperature,
                            },
                        },
                    }
                )
            )
            created = self._receive(cancelled, 300)
            if created.type != "session.created":
                raise RuntimeError(f"MiniCPM initialization failed: {created.reason}.")
            self.setup_seconds = time.perf_counter() - start
        except BaseException:
            self.close()
            raise

    def _receive(
        self, cancelled: Callable[[], bool], timeout: float = 180
    ) -> NativeMessage:
        connection = self._socket
        if connection is None:
            raise RuntimeError("MiniCPM session is not active.")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cancelled():
                raise InterruptedError("MiniCPM session cancelled.")
            try:
                raw = connection.recv(timeout=0.2)
            except TimeoutError:
                continue
            return NativeMessage.model_validate_json(raw)
        raise TimeoutError("MiniCPM native response timed out.")

    def _consume(
        self,
        emit: Callable[[Event], None],
        cancelled: Callable[[], bool],
        *,
        flush: bool = False,
    ) -> None:
        beginning = time.perf_counter()
        try:
            while True:
                message = self._receive(cancelled)
                if message.type == "session.closed":
                    raise RuntimeError(f"MiniCPM session closed: {message.reason}.")
                if message.type == "response.output.delta":
                    if message.kind == "text":
                        emit(Event("text", text=message.text))
                    elif message.kind == "listen":
                        emit(Event("listen"))
                    elif message.kind == "audio" and message.audio is not None:
                        pcm = decode_audio(message.audio)
                        self._output += len(pcm) / (OUTPUT_RATE * 2)
                        if self._output > 240:
                            raise ValueError("MiniCPM output exceeded 240 seconds.")
                        if self._first is None:
                            self._first = (
                                self._elapsed + time.perf_counter() - beginning
                            )
                        emit(Event("audio", pcm=pcm))
                    else:
                        raise ValueError("Invalid MiniCPM output delta.")
                elif message.type == "response.done" and not flush:
                    if self._request and self._request.mode == "turn_based":
                        self._completed = message.reason == "turn_end"
                        if self._completed and self._output == 0:
                            self._completed = False
                            raise RuntimeError(
                                "MiniCPM returned no speech for this turn."
                            )
                    return
                elif message.type == "response.flush.done" and flush:
                    if message.drained is not True:
                        raise RuntimeError(
                            "MiniCPM speech decoder failed or did not finish draining."
                        )
                    return
        finally:
            self._elapsed += time.perf_counter() - beginning

    def push(
        self,
        pcm: bytes,
        emit: Callable[[Event], None],
        cancelled: Callable[[], bool] = never_cancel,
    ) -> None:
        if len(pcm) != FRAME_BYTES or self._input + 1 > MAX_SECONDS:
            raise ValueError(
                "MiniCPM duplex expects one second of mono PCM16 at 16 kHz, "
                "at most 120 frames."
            )
        if (
            self._socket is None
            or self._request is None
            or self._request.mode != "full_duplex"
        ):
            raise RuntimeError("MiniCPM duplex session is not active.")
        self._socket.send(
            json.dumps({"type": "input.append", "input": {"audio": encode_audio(pcm)}})
        )
        self._input += 1
        self._consume(emit, cancelled)

    def turn(
        self,
        emit: Callable[[Event], None],
        cancelled: Callable[[], bool] = never_cancel,
    ) -> None:
        request = self._request
        if self._socket is None or request is None or request.mode != "turn_based":
            raise RuntimeError("MiniCPM turn-based session is not active.")
        content: list[dict[str, str]] = []
        if request.text:
            content.append({"type": "text", "text": request.text})
        if request.audio:
            content.append({"type": "audio", "data": reference_encoded(request.audio)})
            self._input = request.audio.duration_seconds
        messages: list[dict[str, object]] = [
            {"role": "system", "content": request.prompt}
        ]
        messages.extend(
            {"role": turn.role, "content": turn.text} for turn in request.history
        )
        messages.append({"role": "user", "content": content})
        self._socket.send(
            json.dumps(
                {
                    "type": "input.append",
                    "input": {
                        "messages": messages,
                        "streaming": True,
                        "tts": {"enabled": True},
                        "generation": {
                            "max_new_tokens": request.max_tokens,
                            "length_penalty": request.length_penalty,
                        },
                    },
                }
            )
        )
        self._consume(emit, cancelled)

    def finish(
        self,
        emit: Callable[[Event], None],
        cancelled: Callable[[], bool] = never_cancel,
    ) -> None:
        if self._socket is None:
            raise RuntimeError("MiniCPM session is not active.")
        self._socket.send('{"type":"input.flush"}')
        self._consume(emit, cancelled, flush=True)

    def metrics(self) -> Metrics:
        basis = (
            self._input
            if self._request and self._request.mode == "full_duplex"
            else self._output
        )
        return Metrics(
            self._input,
            self._output,
            self._elapsed,
            self._elapsed / basis if basis else None,
            self._first,
            self._completed,
        )

    def close(self) -> None:
        connection, self._socket = self._socket, None
        try:
            if connection is not None:
                connection.close()
        finally:
            self._process.close()
