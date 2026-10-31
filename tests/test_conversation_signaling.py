"""Browser ICE messages reach only the reserved peer, without loading models."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from voicelab.conversations.contracts import Backend
from voicelab_web.conversations.routes import Conversations, install
from voicelab_web.conversations.session import ConversationSession
from voicelab_web.conversations.transport.schemas import IceCandidate, IceUpdate
from voicelab_web.transport.lease import SessionLease


class IceTests(unittest.TestCase):
    def test_browser_patch_is_validated_and_forwarded(self) -> None:
        async def release() -> None:
            pass

        def unused(request: object) -> Backend:
            raise AssertionError("Signaling must not load inference")

        manager = Conversations(SessionLease(), release, unused)
        app = FastAPI()
        install(app, manager)
        session = MagicMock(spec=ConversationSession)
        session.update_ice = AsyncMock()
        with TestClient(app) as client:
            # Create the reservation on the application's event loop.
            with (
                patch("platform.system", return_value="Darwin"),
                patch("platform.machine", return_value="arm64"),
            ):
                response = client.post(
                    "/api/conversations",
                    headers={"origin": "http://testserver"},
                    json={"model": "kokoro"},
                )
            identity = response.json()["session_id"]
            reservation = manager.get(identity)
            reservation.session = session
            endpoint = f"/api/conversations/{identity}/offer"
            payload = {
                "pc_id": "peer",
                "candidates": [{"candidate": "", "sdp_mid": "0", "sdp_mline_index": 0}],
            }
            self.assertEqual(client.patch(endpoint, json=payload).status_code, 403)
            headers = {"origin": "http://testserver"}
            self.assertEqual(
                client.patch(endpoint, headers=headers, json={}).status_code, 422
            )
            self.assertEqual(
                client.patch(endpoint, headers=headers, json=payload).status_code,
                204,
            )
            session.update_ice.assert_awaited_once_with(
                IceUpdate.model_validate(payload)
            )
            reservation.session = None
            self.assertEqual(
                client.patch(endpoint, headers=headers, json=payload).status_code,
                409,
            )
            client.delete(f"/api/conversations/{identity}", headers=headers)


class PeerTests(unittest.IsolatedAsyncioTestCase):
    async def test_candidates_and_end_marker_are_bound_to_the_peer(self) -> None:
        backend = MagicMock(spec=Backend)
        backend.input_rate = 16000
        backend.frame_bytes = 2560
        backend.duplex = True
        session = ConversationSession(backend)
        try:
            with patch.object(
                session.connection, "add_ice_candidate", new_callable=AsyncMock
            ) as add:
                update = IceUpdate(
                    pc_id=session.connection.pc_id,
                    candidates=[
                        IceCandidate(
                            candidate="candidate:1 1 UDP 1 127.0.0.1 9000 typ host",
                            sdp_mid="0",
                            sdp_mline_index=0,
                        ),
                        IceCandidate(candidate="", sdp_mid="0", sdp_mline_index=0),
                    ],
                )
                await session.update_ice(update)
                self.assertEqual(add.await_count, 2)
                candidate = add.await_args_list[0].args[0]
                self.assertEqual(candidate.ip, "127.0.0.1")
                self.assertEqual(candidate.sdpMid, "0")
                self.assertIsNone(add.await_args_list[1].args[0])
                with self.assertRaisesRegex(ValueError, "another peer"):
                    await session.update_ice(
                        update.model_copy(update={"pc_id": "other"})
                    )
                self.assertEqual(add.await_count, 2)
            self.assertIs(session.conversation.worker.rtvi, session.rtvi)
        finally:
            await session.close()
