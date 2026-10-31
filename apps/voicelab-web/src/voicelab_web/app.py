"""One FastAPI application composing model-specific transports and a shared UI."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from voicelab_models.breeze.synthesis import SynthesisRequest as BreezeRequest
from voicelab_models.sesame.model import SynthesisRequest as SesameRequest
from voicelab_web.catalog import ModelInfoUnion, models
from voicelab_web.conversations.composition import create_backend
from voicelab_web.conversations.routes import Conversations
from voicelab_web.conversations.routes import install as install_pipecat
from voicelab_web.models.conversation_routes import install as install_conversations
from voicelab_web.models.nemotron.engine import Engine
from voicelab_web.models.nemotron.session import serve_session as serve_nemotron
from voicelab_web.models.nemotron.tool_session import serve_tool_session
from voicelab_web.models.synthesis.composition import install as install_synthesis
from voicelab_web.models.synthesis.routes import SynthesisRoutes
from voicelab_web.models.synthesis.service import SynthesisService, Synthesizer
from voicelab_web.schemas import ServerConfig
from voicelab_web.transport.lease import SessionLease


def create_app(
    config: ServerConfig,
    engine: Engine | None = None,
    sesame: SynthesisService[SesameRequest] | None = None,
    breeze: SynthesisService[BreezeRequest] | None = None,
) -> FastAPI:
    owned_engine = engine if engine is not None else Engine()

    def create_sesame_backend() -> Synthesizer[SesameRequest]:
        from voicelab_inference.sesame.torch.runtime import ResidentCSM

        return ResidentCSM(config.sesame_device, config.sesame_engine)

    def create_breeze_backend() -> Synthesizer[BreezeRequest]:
        from voicelab_inference.breeze.mlx.runtime import BreezeMLX

        return BreezeMLX()

    synthesis = (
        sesame
        if sesame is not None
        else SynthesisService(
            SesameRequest,
            create_sesame_backend,
        )
    )
    breeze_synthesis = (
        breeze
        if breeze is not None
        else SynthesisService(
            BreezeRequest,
            create_breeze_backend,
        )
    )
    lease = SessionLease()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        try:
            yield
        finally:
            await conversations.close()
            await synthesis_routes.release()
            await owned_engine.call(owned_engine.unload)
            owned_engine.shutdown()

    app = FastAPI(title="Voice Lab", lifespan=lifespan, docs_url=None, redoc_url=None)

    async def release_duplex() -> None:
        await owned_engine.call(owned_engine.unload)

    synthesis_routes = SynthesisRoutes(app, lease, release_duplex)
    synthesis_routes.add("sesame", synthesis)
    synthesis_routes.add("breeze", breeze_synthesis)
    install_synthesis(synthesis_routes, config)

    async def release_local() -> None:
        await synthesis_routes.release()
        await release_duplex()

    install_conversations(app, lease, release_local)
    conversations = Conversations(
        lease,
        release_local,
        lambda request: create_backend(request, synthesis_routes.voices),
    )
    install_pipecat(app, conversations)
    static = Path(__file__).parent / "static"

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static / "index.html")

    @app.get("/api/models", response_model=list[ModelInfoUnion])
    async def catalog() -> tuple[ModelInfoUnion, ...]:
        return models(config)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "active_model": lease.active}

    if config.ca_file is not None:
        ca_path = Path(config.ca_file)

        @app.get("/local-ca.cer")
        async def local_ca() -> FileResponse:
            return FileResponse(
                ca_path,
                media_type="application/x-x509-ca-cert",
                filename="VoiceLab-local-CA.cer",
            )

    @app.websocket("/api/nemotron/session")
    async def nemotron_session(socket: WebSocket) -> None:
        async def run() -> None:
            await synthesis_routes.release()
            await serve_nemotron(socket, owned_engine)

        await lease.run(socket, "nemotron", run)

    @app.websocket("/api/nemotron/tools")
    async def nemotron_tools(socket: WebSocket) -> None:
        async def run() -> None:
            await synthesis_routes.release()
            await serve_tool_session(socket, owned_engine)

        await lease.run(socket, "nemotron", run)

    app.mount("/static", StaticFiles(directory=static), name="static")
    return app
