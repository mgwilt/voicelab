# Extend the CLI

Read [command usage](../usage/cli.md) before a command change.
The checkout command is `./voicelab`. It starts the shared website when no subcommand is supplied.

## Select the owner

- `voicelab.launch.cli`: argument routing and explicit model/runtime commands.
- `voicelab.launch.workspace`: environment preparation, asset builds, and process replacement.
- `voicelab.<model>.cli`: model-specific arguments and runtime construction.
- `voicelab.<model>.experiments`: experiment execution and reports.

Keep the launcher dependent on the standard library only.
Root help and model listing must not install inference dependencies or load weights.

## Add a command

1. Add a concrete runtime command to its model's command tuple.
2. Keep model inference under `voicelab_inference.<model>.<runtime>`.
3. Validate model inputs before loading weights.
4. Test argument forwarding, unsupported runtimes, and failure behavior.
5. Update the model guide.

A new workflow can have different arguments or a different subcommand.
Do not force all models into one request format or generation method.
Add a notebook, service, or other entry point only when the workflow needs it.
Do not add automatic module scanning, plugin registries, or one application per model.

## Preserve setup behavior

Setup uses the root workspace and locked dependencies.
The launcher rebuilds frontend assets when sources change or required assets are missing.
It writes an asset fingerprint only after a successful build.
A failed preparation step must stop startup.

Keep CLI parsing, experiment policy, and report presentation separate.
Inject runtime protocols into experiment logic.
Retain immutable run directories and failed reports as described in [measurements](../usage/measurements.md).
