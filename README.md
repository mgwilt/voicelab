# Voice Lab

Test voice AI models. Compare audio quality, latency, and real-time factor (RTF) across runtimes and devices.

| Model | Test |
| --- | --- |
| [Nemotron VoiceChat](docs/models/nemotron.md) | Live voice conversation |
| [Sesame CSM-1B](docs/models/sesame.md) | Speech synthesis with optional reference audio |
| [Breeze TTS](docs/models/breeze.md) | Speech synthesis controlled by voice instructions |

## Start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Bun](https://bun.sh/docs/installation).
Run this command from the repository root:

```sh
./voicelab
```

The launcher prepares the lab automatically. Open `http://127.0.0.1:8443` and select a model.
For mobile access, read the [usage guide](docs/usage/README.md).

## Run an experiment

```sh
./voicelab models
./voicelab run breeze --text 'Hello'
```

| Option | Applies to | Use |
| --- | --- | --- |
| `--text` | Breeze, Sesame | Supply the text to speak. |
| `--input` | Nemotron | Supply a mono 16 kHz PCM16 WAV. |
| `--device` | Sesame | Select `cpu`, `mps`, or `cuda`. This option is required. |
| `--runtime` | `run` | Select a runtime listed by `models`. |
| `--output` | `run` | Set the result directory. The default creates a new directory. |
| `--config` | `serve` | Select a local server configuration file. |

Read [CLI usage](docs/usage/cli.md) for more options. Use `./voicelab run <model> --help` for model controls.
Coding agents start at [the documentation index](docs/README.md).
