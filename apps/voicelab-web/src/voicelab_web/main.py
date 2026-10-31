"""Start the local server using an optional ignored TOML configuration."""

import argparse
import tomllib
from pathlib import Path

import uvicorn

from voicelab_web.app import create_app
from voicelab_web.schemas import ServerConfig


class Arguments(argparse.Namespace):
    config: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Lab model testing")
    parser.add_argument("--config", type=Path, default=Path(".local/server.toml"))
    args = parser.parse_args(namespace=Arguments())
    raw = tomllib.loads(args.config.read_text()) if args.config.exists() else {}
    config = ServerConfig.model_validate(raw)
    if config.host not in {"127.0.0.1", "localhost", "::1"} and (
        config.certfile is None or config.keyfile is None
    ):
        parser.error("LAN microphone use requires TLS: configure certfile and keyfile.")
    built = Path(__file__).parent / "static/dist/main.js"
    if not built.is_file():
        parser.error("Build the frontend first: bun run --cwd apps/voicelab-web build")
    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        ssl_certfile=config.certfile,
        ssl_keyfile=config.keyfile,
        ws_max_size=16384,
        ws_max_queue=4,
        proxy_headers=False,
    )
