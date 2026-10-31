"""MiniCPM boundary and native lifecycle checks without model weights."""

import base64
import math
import os
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pydantic import ValidationError

from voicelab_inference.minicpm.native.prepare import run_command
from voicelab_inference.minicpm.native.protocol import NativeMessage, decode_audio
from voicelab_inference.minicpm.native.runtime import MiniCPMNative
from voicelab_models.minicpm.model import HistoryTurn, Request


class MiniCPMTests(unittest.TestCase):
    def test_invalid_requests(self) -> None:
        cases: list[dict[str, object]] = [
            {"mode": "turn_based"},
            {"text": "Not a live frame"},
            {"temperature": math.nan},
            {"max_tokens": 0},
            {"prompt": "  "},
            {"history": (HistoryTurn(role="assistant", text="Wrong order"),)},
        ]
        for fields in cases:
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                Request.model_validate(fields)

    def test_audio_boundary(self) -> None:
        for payload in (
            "?",
            "",
            base64.b64encode(b"x").decode(),
            base64.b64encode(struct.pack("<f", math.nan)).decode(),
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                decode_audio(payload)
        pcm = decode_audio(base64.b64encode(struct.pack("<3f", -2, 0, 2)).decode())
        self.assertEqual(struct.unpack("<3h", pcm), (-32767, 0, 32767))

    def test_completion_requires_audio(self) -> None:
        runtime = MiniCPMNative()
        runtime._request = Request(mode="turn_based", text="Hello")
        with (
            patch.object(
                runtime,
                "_receive",
                return_value=NativeMessage(
                    type="response.done",
                    reason="turn_end",
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "no speech"),
        ):
            runtime._consume(lambda _: None, lambda: False)
        self.assertFalse(runtime.metrics().completed)

    def test_cap_is_not_completion(self) -> None:
        runtime = MiniCPMNative()
        runtime._request = Request(mode="turn_based", text="Hello")
        with patch.object(
            runtime,
            "_receive",
            return_value=NativeMessage(
                type="response.done",
                reason="incomplete",
            ),
        ):
            runtime._consume(lambda _: None, lambda: False)
        self.assertFalse(runtime.metrics().completed)

    def test_invalid_frame_does_not_touch_socket(self) -> None:
        with self.assertRaisesRegex(ValueError, "one second"):
            MiniCPMNative().push(bytes(2), lambda _: None)

    def test_failed_preparation(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp,
            self.assertRaisesRegex(RuntimeError, "preparation failed"),
        ):
            run_command(
                [sys.executable, "-c", "raise SystemExit(7)"],
                Path(temp) / "log",
                lambda: False,
            )

    def test_cancel_kills_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            pidfile = Path(temp) / "child.pid"
            child = (
                "import signal,time; "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
            )
            parent = (
                "import subprocess,sys,time; from pathlib import Path; "
                f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); "
                f"Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(30)"
            )
            cancel = threading.Event()
            timer = threading.Timer(0.5, cancel.set)
            timer.start()
            try:
                with self.assertRaises(InterruptedError):
                    run_command(
                        [sys.executable, "-c", parent],
                        Path(temp) / "log",
                        cancel.is_set,
                    )
            finally:
                timer.cancel()
            pid = int(pidfile.read_text())
            for _ in range(30):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                self.fail("Preparation left a descendant alive after cancellation.")

    def test_close_releases_process_if_socket_fails(self) -> None:
        runtime = MiniCPMNative()
        socket = Mock()
        socket.close.side_effect = RuntimeError("socket failure")
        process = Mock()
        runtime._socket = socket
        runtime._process = process
        with self.assertRaisesRegex(RuntimeError, "socket failure"):
            runtime.close()
        process.close.assert_called_once()
