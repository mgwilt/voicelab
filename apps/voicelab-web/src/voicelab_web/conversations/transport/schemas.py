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


class Created(BaseModel):
    session_id: str
    offer_url: str
