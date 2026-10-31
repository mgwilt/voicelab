"""One FastAPI application composing model-specific transports and a shared UI."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from voicelab_models.breeze.synthesis import SynthesisRequest as BreezeRequest
from voicelab_models.sesame.model import SynthesisRequest as SesameRequest
from voicelab_web.catalog import ModelInfoUnion, models
from voicelab_web.models.nemotron.engine import Engine
from voicelab_web.models.nemotron.session import serve_session as serve_nemotron
from voicelab_web.models.synthesis.service import SynthesisService, Synthesizer
from voicelab_web.models.synthesis.session import serve_session as serve_synthesis
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
            await owned_engine.call(owned_engine.close_session)
            owned_engine.shutdown()

    app = FastAPI(title="Voice Lab", lifespan=lifespan, docs_url=None, redoc_url=None)
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
        await lease.run(
            socket, "nemotron", partial(serve_nemotron, socket, owned_engine)
        )

    @app.websocket("/api/sesame/session")
    async def sesame_session(socket: WebSocket) -> None:
        await lease.run(socket, "sesame", partial(serve_synthesis, socket, synthesis))

    @app.websocket("/api/breeze/session")
    async def breeze_session(socket: WebSocket) -> None:
        await lease.run(
            socket, "breeze", partial(serve_synthesis, socket, breeze_synthesis)
        )

    app.mount("/static", StaticFiles(directory=static), name="static")
    return app
