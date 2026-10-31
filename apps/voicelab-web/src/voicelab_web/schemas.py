"""Portable server configuration; personal values belong in .local/."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from voicelab_models.sesame.model import Device, Engine


class ServerConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    host: str = "127.0.0.1"
    port: int = Field(default=8443, ge=1024, le=65535)
    certfile: str | None = None
    keyfile: str | None = None
    ca_file: str | None = None

    sesame_device: Device = "cpu"
    sesame_engine: Engine = "direct-depth"
    phonon_device: Literal["cpu", "mps", "cuda"] = "cpu"
    omnivoice_device: Literal["cpu", "mps", "cuda"] = "cpu"
    kitten_device: Literal["cpu", "mps", "cuda"] = "cpu"
