# Organize the code

Voice Lab has three library packages and two applications.
A package boundary represents an installation, dependency, or release requirement.
A model or device does not require a new package.

## Keep dependencies inward

```text
applications → inference → model definitions → core
```

Applications can also use model definitions and core directly.
Core uses only the standard library.
Model definitions do not import inference frameworks.
Inference code does not import applications or another model's implementation.

## Select the owner

- `apps/voicelab`: commands, experiment execution, and reports.
- `apps/voicelab-web`: one FastAPI website and its frontend.
- `packages/voicelab-core`: framework-free contracts, audio values, and shared measurements.
- `packages/voicelab-models`: model identities, constraints, and validated requests.
- `packages/voicelab-inference`: model loading, caches, decoding, and device work.

Use the same module pattern for each model:

```text
voicelab_models.<model>
voicelab_inference.<model>.<runtime>
voicelab.<model>
```

The current inference modules are `nemotron.mlx`, `sesame.torch`, and `breeze.mlx`.
Keep model identity separate from runtime and device support.
Add a runtime module only when an implementation exists.
Do not create placeholder CUDA, TPU, or ANE directories.

## Compose workflows

Applications select and construct runtime objects. Pass those objects to experiment logic through narrow protocols.
Keep framework details inside the runtime that uses them.
Separate vendor adaptation from custom sampling, decoding, and kernels.

Share a contract only when its meaning matches in more than one workflow.
The website shares synthesis transport and timing values. Duplex conversation uses a separate transport contract.
One website session owns inference until device work and cleanup finish.

Keep one root uv workspace, environment, and lockfile. Keep platform extras in the inference package.
Use earlier projects as algorithm and evidence references. Adapt their code to this structure and retain attribution.
Do not import their separate applications, sites, environments, or incidental abstractions.

Module dependency tests enforce these boundaries. Read [verification](verification.md) before a change is complete.
