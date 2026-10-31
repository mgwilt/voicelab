# Run commands

Run commands from the repository root. Install uv and Bun before the first launch.

## Start and inspect

```sh
./voicelab
./voicelab models
./voicelab --help
```

The first command starts the shared website. The other commands show available models and command help.
Root help and model listing do not prepare the Python environment.

## Save an experiment

Use this command form:

```text
./voicelab run <model> [model options]
```

For example:

```sh
./voicelab run breeze --text 'Hello there.'
./voicelab run breeze --help
```

Each model has its own input options. Read its [model guide](../models/README.md) before a measured run.

| Option | Use |
| --- | --- |
| `--runtime` | Select a runtime listed by `models`. |
| `--output` | Set a new directory for reports and audio. |
| `--help` | Show options for the selected command or model. |

Without `--output`, the launcher creates a new directory under `artifacts/runs/`.
An unknown runtime causes an error before setup. A model can reject unsupported inputs or settings.

## Select server settings

```sh
./voicelab serve --config .local/server.toml
```

The default configuration file is `.local/server.toml`.
Read [network access](network.md) for server settings. Read the model guide for device and engine settings.

To change command dispatch or setup behavior, read [CLI development](../development/cli.md).
