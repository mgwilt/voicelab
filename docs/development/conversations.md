# Change conversation orchestration

Read [architecture](architecture.md) and [frontend](frontend.md) first.

## Keep inference independent

Pipecat belongs to the web application. Do not import it into library packages.

The application separates these responsibilities:

- `conversations/routes.py`: reservations, same-origin checks, signaling, and lease ownership.
- `conversations/session.py`: peer connection and Pipecat pipeline lifetime.
- `conversations/processor.py`: bounded PCM delivery, interruptions, and serialized device work.
- `conversations/adapters/`: model-specific conversation composition.
- `frontend/conversation/`: validated events, client resources, settings, and conversation presentation.

Use the existing synthesis services for voices. Do not add another model registry or inference implementation.
New recognition and text runtimes follow the existing model/runtime package pattern.

## Preserve each paradigm

Cascade uses managed local transcription, text generation, and a selected synthesis service.
Turn-based adapters keep native audio requests and bounded history.
Duplex adapters keep native sample rates, frame sizes, caches, and turn-taking.
Do not apply external VAD gating to duplex input.

A session owns all required models until cleanup completes.
Cancellation must wait for uninterruptible device work. Stale output must not reach the client.
Stage completed exchanges, then commit them after the output transport drains.
Discard interrupted exchanges. Server drain is not proof of physical speaker playback.

## Inspect Pipecat source

The web package pins Pipecat v1.14.0 to Git commit `82c4fdb484db32e3ef933d03c59a395dd7f9a53b`.
uv fetches it and installs readable Python source. The repository contains no Pipecat checkout or submodule.

Locate the installed source:

```sh
uv run --all-packages --extra mlx --locked python -c \
  'import importlib.util; print(importlib.util.find_spec("pipecat").origin)'
```

Use the installed version as the API reference. Do not assume documentation for `main` matches the pin.
Keep full upstream inspection checkouts under ignored `.local/sources/` if needed.

Pipecat constrains SoundFile to 0.13.x and soxr to 1.0.x.
The workspace uses those compatible versions without dependency overrides.
Recheck audio decoding and resampling when updating them.

The browser transport is pinned to 1.10.7 with client 1.13.0.
Later transport packages reference an unpublished type dependency.
Keep `skipLibCheck` disabled. Verify published declarations before upgrading.

## Verify changes

Run `make check`. Exercise real WebRTC input, output, interruption, and teardown separately.
Test real model runs on the claimed runtime. Keep recordings and reports in ignored artifacts.
Physical mobile browser testing remains separate from loopback transport and DOM tests.

## Verified integration paths

Real Apple Silicon runs exercised all 15 voice adapters with repeated input.
MiniCPM was exercised in both turn-based and duplex modes.
Local WebRTC tests exercised the Whisper–Qwen–Kokoro cascade, Qwen Omni turns, and Nemotron duplex.
They verified returned audio and cleanup. Turn-based tests also verified history after server playback drain.
These checks do not establish physical mobile playback, subjective quality, or runtime parity.
