# Change the website

One FastAPI application serves all model tests. The frontend uses vanilla TypeScript 7 components.
Bun installs dependencies, builds browser files, and runs frontend tests.

## Keep the workbench mobile first

Use a compact workbench with clear navigation, tabs, panels, and status.
Adapt these elements for touch and narrow screens. Keep controls usable with the software keyboard open.
Do not require hover, horizontal page scrolling, or a desktop sidebar to run an experiment.

The shell owns navigation and run actions. Model panels own typed experiment controls.
Changing views must preserve control values and active session ownership.
Keep layout independent of how configuration is edited. Monaco, declarative configuration, and a desktop mode are deferred.

## Select the owner

Paths below are relative to `apps/voicelab-web`.

- `frontend/domain/`: model types, session events, and run state; no DOM imports.
- `frontend/components/`: model selection, workbench navigation, run controls, measurements, and transcripts.
- `frontend/models/`: model forms, request construction, and explicit panel selection.
- `frontend/models/synthesis/`: shared synthesis session and response decoding.
- `frontend/transport/`: catalog decoding and WebSocket lifetime.
- `frontend/audio/`: permission, resampling, capture worklet, and playback.
- `src/voicelab_web/models/`: duplex transport and shared synthesis transport.
- `src/voicelab_web/transport/`: inference ownership and PCM16 wire conversion.
- `src/voicelab_web/catalog.py`: model capabilities and route identities.

Components receive typed data and callbacks. Shared components do not import model implementations.
The application constructs runtimes from `voicelab_inference` and requests from `voicelab_models`.
Synthesis uses `synthesize(request, emit) -> SynthesisMetrics`, with a distinct request type for each model.
Keep duplex transport separate when its behavior differs.

## Validate boundaries

Treat incoming JSON as `unknown`. Decode it into discriminated unions before use.
Reject unknown variants, invalid sample rates, non-finite metrics, and inconsistent completion states.
Pydantic validates server requests independently.

Use exhaustive branches and strict TypeScript checks.
Do not use `any`, unsafe casts, non-null assertions, suppression comments, or implicit optional-field conversion.
Keep nullability, indexed access, returns, overrides, and unused-symbol checks enabled.
Bun transpilation does not replace the TypeScript check.

## Own session resources

Each session owns its microphone, audio context, socket, timers, and cancellation.
Release listeners when a component is disposed.
Stop pending work on cancellation, model changes, and page exit.
Release tracks if microphone permission arrives after cancellation.
Ignore callbacks from an old run.

Let playback drain after normal completion. Stop playback immediately on cancellation.
Bound server queues, browser queues, and session duration.
The server retains inference ownership until device work and cleanup finish, including after disconnects.
Keep input RTF and output RTF labels distinct.

## Add a model panel

1. Add typed model requests and a concrete runtime in the existing packages.
2. Construct the runtime in its application session or backend owner.
3. Add the model descriptor to the server catalog.
4. Register its server route. Extend the frontend model union, catalog decoder, and panel selection.
5. Add controls only for that model's distinct inputs.
6. Test decoding, cancellation, resource cleanup, and real device behavior.

Reuse synthesis transport only when its semantics match. Do not create another site or a universal request dictionary.

## Build and check

```sh
bun install --cwd apps/voicelab-web --frozen-lockfile
bun run --cwd apps/voicelab-web check
bun run --cwd apps/voicelab-web build
bun run --cwd apps/voicelab-web test
```

The frontend lockfile is `bun.lock`. FastAPI serves generated `static/dist/` files.
Those files are ignored and rebuilt from source.
For required checks and browser acceptance limits, read [verification](verification.md).
