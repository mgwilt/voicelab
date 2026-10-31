# Organize the code

Voice Lab has three library packages and two applications.
A package boundary represents an installation, dependency, or release requirement.
A model or device does not require a new package.

## Keep dependencies inward

```text
applications → inference → model definitions → core
```

Applications can also use model definitions and core directly.
The web host uses `voicelab.conversations` for shared conversation orchestration.
That application module must not import the web host or HTTP/WebRTC libraries.
Core uses only the standard library.
Model definitions do not import inference frameworks.
Inference code does not import applications or another model's implementation.

## Select the owner

- `apps/voicelab`: commands, run execution, reports, and reusable conversation pipelines.
- `apps/voicelab-web`: one FastAPI website and its frontend.
- `packages/voicelab-core`: framework-free contracts, audio values, and shared measurements.
- `packages/voicelab-models`: model identities, constraints, and validated requests.
- `packages/voicelab-inference`: model loading, caches, decoding, and device work.

Use the same module pattern for each model:

```text
voicelab_models.<model>
voicelab_inference.<model>.<runtime>
voicelab.models.<model>
```

The CLI application separates startup, model commands, and shared run support:

```text
voicelab/
  launch/             # Startup, command dispatch, and device arguments
  models/<model>/     # Model commands and run orchestration
  runs/               # Shared evidence and execution support
    synthesis/        # Lifecycle for compatible synthesis workflows
```

Model commands can use shared run support. Shared run support must not import model commands or concrete runtimes.
Model commands must not import another model's application modules.
Keep startup dependent on the standard library. It dispatches commands without loading model modules.

Use **run** for an execution whose outputs and status are retained.
Use **session** for an interactive connection and its resource lifetime.
Neither term requires a benchmark or evaluation workflow.

Examples of inference modules are `nemotron.mlx`, `sesame.torch`, and `breeze.mlx`.
Keep model identity separate from runtime and device support.
Shared reference-audio schemas live in `voicelab_models.audio`.
Decoding, resampling, and temporary reference files live in `voicelab_inference.audio`.
These modules must not import a model implementation.
Add a runtime module only when an implementation exists.
Do not create placeholder CUDA, TPU, or ANE directories.

## Compose workflows

Applications select and construct runtime objects. Pass those objects to run logic through narrow protocols.
Keep framework details inside the runtime that uses them.
Separate vendor adaptation from custom sampling, decoding, and kernels.

Share a contract only when its meaning matches in more than one workflow.
The website shares synthesis transport and timing values. Duplex conversation uses a separate transport contract.
One website session owns inference until device work and cleanup finish.

Keep one root uv workspace, environment, and lockfile. Keep platform extras in the inference package.
Use earlier projects as algorithm and evidence references. Adapt their code to this structure and retain attribution.
An upstream dependency that needs a packaging patch can live under `third_party/`.
Preserve its source, licenses, revision, and a list of changes. Keep lab adapters outside that directory.
Do not import their separate applications, sites, environments, or incidental abstractions.

Module dependency tests enforce these boundaries. Read [verification](verification.md) before a change is complete.
