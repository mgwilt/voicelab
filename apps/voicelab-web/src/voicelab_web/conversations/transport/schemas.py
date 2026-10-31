"""Strict WebRTC signaling boundaries."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Offer(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    sdp: str = Field(min_length=1, max_length=100000)
    type: Literal["offer"]
    pc_id: str | None = None
    restart_pc: bool = False


class Answer(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    sdp: str
    type: Literal["answer"]
    pc_id: str


class IceCandidate(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    candidate: str = Field(max_length=4096)
    sdp_mid: str = Field(max_length=256)
    sdp_mline_index: int = Field(ge=0, le=65535)


class IceUpdate(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    pc_id: str = Field(min_length=1, max_length=256)
    candidates: list[IceCandidate] = Field(max_length=64)


class Created(BaseModel):
    session_id: str
    offer_url: str
