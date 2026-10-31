# Change conversation orchestration

Read [architecture](architecture.md) and [frontend](frontend.md) first.

## Keep inference independent

Pipecat orchestration belongs to `apps/voicelab/src/voicelab/conversations/`.
Do not import it into model, inference, or core packages.

The application separates these responsibilities:

- `voicelab/conversations/pipeline.py`: transport-independent pipeline construction and lifetime.
- `voicelab/conversations/services/`: separate STT, LLM, and TTS services; call state and cancellation.
- `voicelab/conversations/processor.py` and `adapters/`: native audio turns and duplex models.
- `voicelab_web/conversations/`: HTTP reservations, WebRTC signaling, and web service composition.
- `apps/voicelab-web/frontend/conversation/`: browser resources and presentation.

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

## Use framework lifecycle hooks

Pass the RTVI processor to `PipelineWorker`. Let Pipecat install its matching observer.
Accept both POST offers and PATCH ICE updates. Validate the origin and reserved peer before forwarding candidates.
An empty candidate marks the end of candidate gathering.

Keep one browser SDK owner across calls. Its media manager shares a browser media object.
Old callbacks must never stop tracks delivered to a later call.
Release microphone tracks after late permission completes. Do not wait for bot-ready to finish a cancelled call.

Conversation capture uses `frontend/audio/processing.ts` to configure the SDK's local microphone track.
Apply processing before signaling, and repeat it for replacement microphone tracks.
Preserve existing device constraints. Read effective values with `getSettings()`, not `getConstraints()`.
Keep disabled, unreported, and rejected settings distinct. Do not expose device IDs in diagnostics.
Ignore late results after track replacement or call closure. Retain each connection attempt's abort signal across awaits.
Render microphone warnings separately from transient call status. Keep remote audio on its WebRTC-backed audio element.
`SpeakerControl` owns the playback mute preference and its `volumechange` listener.
Set the audio element's `muted` property; do not pause playback or change capture and transport for speaker mute.
Retain the preference across calls in the view. Detach listeners on disconnect and disposal.
Browser settings and mocked tracks do not prove acoustic echo removal; verify speaker playback on the target device.
`frontend/audio/call-session.ts` requests `play-and-record` before microphone capture when the browser exposes the Audio Session API.
The call owns this setting until microphone cleanup finishes, including late permission and failed SDK disconnects.
Restore the previous setting on close. Do not overwrite a later audio mode chosen by another page feature.
Report unavailable or rejected routing separately from microphone processing. This setting does not implement an additional echo canceller.
See the [browser constraint API](https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamTrack/applyConstraints)
and [effective settings API](https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamTrack/getSettings).
For call routing, read the [Audio Session API](https://developer.mozilla.org/en-US/docs/Web/API/Audio_Session_API).

The cascade uses `SegmentedSTTService`, `LLMUserAggregator`, `LLMService`, and `TTSService`.
The user aggregator owns Silero VAD and Smart Turn. Whisper consumes raw PCM segments.
Qwen emits response frames. One complete text frame reaches TTS to preserve each voice's synthesis behavior.
The TTS service owns Pipecat audio contexts. Conversation history commits after the output transport drains.

All local services share one serialized device worker. Bound pending speech segments and output events.
Cancellation invalidates the turn before waiting for device work. Discard stale transcription, text, and audio.
Native duplex clocks and state remain in their adapter; they do not pass through cascade services.

`ConversationPipeline` accepts input/output frame processors and an event callback.
A host can use it without FastAPI, HTTP, or WebRTC. The web host supplies those transports separately.
Install the `voicelab[conversations]` extra when using this application interface.

## Preserve transcript semantics

Text notices declare `text_mode`: `delta` appends text; `snapshot` replaces the current speaker's text for that turn.
A snapshot can revise earlier characters and does not imply a completed utterance.
`final` marks completion and carries the complete text, regardless of update mode.
Nemotron's runtime events retain cumulative `text` despite their `_delta` event names; its conversation adapter sends snapshots.
Do not deduplicate text by matching prefixes. Genuine repeated words must survive delta updates.

## Inspect Pipecat source

The workspace pins Pipecat v1.14.0 to Git commit `82c4fdb484db32e3ef933d03c59a395dd7f9a53b`.
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
After the application extraction, WebRTC checks repeated the cascade, Qwen Omni, and Nemotron paths.
Headless pipeline tests cover whole-response synthesis, sample rates, bounded history, interruption recovery, and cleanup before signaling.
These checks do not establish physical mobile playback, subjective quality, or runtime parity.
