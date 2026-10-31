# Voice Lab

Test voice AI models. Compare audio quality, latency, and real-time factor (RTF) across runtimes and devices.

Local lab features: ☑ exposed · ☐ not exposed. Unchecked features can still exist in the upstream model.
TTS = text to speech. S2S = speech to speech.

| Model | TTS | S2S | Duplex | Stream | Clone | Prompt | Presets |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [Nemotron VoiceChat](docs/models/conversation/nemotron.md) | ☐ | ☑ | ☑ | ☑ | ☐ | ☐ | ☐ |
| [Sesame CSM-1B](docs/models/synthesis/sesame.md) | ☑ | ☐ | ☐ | ☑ | ☑ | ☐ | ☐ |
| [Breeze TTS](docs/models/synthesis/breeze.md) | ☑ | ☐ | ☐ | ☑ | ☑ | ☑ | ☐ |
| [Fish Audio Drama 3](docs/models/synthesis/drama.md) | ☑ | ☐ | ☐ | ☑ | ☑ | ☑ | ☐ |
| [Maya 2 Native](docs/models/synthesis/maya.md) | ☑ | ☐ | ☐ | ☑ | ☑ | ☐ | ☑ |
| [Audio8 Realistic 0.6B](docs/models/synthesis/audio8.md) | ☑ | ☐ | ☐ | ☐ | ☑ | ☐ | ☐ |
| [Ming 3.0 TTS 0.5B](docs/models/synthesis/ming_tts.md) | ☑ | ☐ | ☐ | ☐ | ☑ | ☑ | ☑ |
| [Phonon 31M](docs/models/synthesis/phonon.md) | ☑ | ☐ | ☐ | ☑ | ☐ | ☐ | ☑ |
| [OmniVoice](docs/models/synthesis/omnivoice.md) | ☑ | ☐ | ☐ | ☐ | ☑ | ☑ | ☐ |
| [Kokoro 82M](docs/models/synthesis/kokoro.md) | ☑ | ☐ | ☐ | ☑ | ☐ | ☐ | ☑ |
| [KittenTTS 2](docs/models/synthesis/kitten.md) | ☑ | ☐ | ☐ | ☑ | ☑ | ☑ | ☑ |
| [PersonaPlex 7B](docs/models/conversation/personaplex.md) | ☐ | ☑ | ☑ | ☑ | ☑ | ☐ | ☑ |
| [Ming 3.0 Audio Tiny](docs/models/conversation/ming_audio.md) | ☑ | ☑ | ☐ | ☐ | ☑ | ☐ | ☑ |

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
| `--text` | Speech synthesis | Supply the text to speak. |
| `--input` | Audio workflows | Supply a WAV in the model’s required format. |
| `--device` | PyTorch workflows | Select a supported device. |
| `--runtime` | `run` | Select a runtime listed by `models`. |
| `--output` | `run` | Set the result directory. The default creates a new directory. |
| `--config` | `serve` | Select a local server configuration file. |

Read [CLI usage](docs/usage/cli.md) for more options. Use `./voicelab run <model> --help` for model controls.
Coding agents start at [the documentation index](docs/README.md).

## Model licenses

| Model | Weights license |
| --- | --- |
| [Nemotron VoiceChat](docs/models/conversation/nemotron.md) | [OpenMDW 1.1](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B#licenseterms-of-use) |
| [Sesame CSM-1B](docs/models/synthesis/sesame.md) | [Apache-2.0](https://huggingface.co/sesame/csm-1b) |
| [Breeze TTS](docs/models/synthesis/breeze.md) | [BreezeBlue Research and Non-Commercial](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE) |
| [Fish Audio Drama 3](docs/models/synthesis/drama.md) | [Fish Audio Community](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE) |
| [Maya 2 Native](docs/models/synthesis/maya.md) | [Apache-2.0](https://huggingface.co/maya-research/maya2-native#license) |
| [Audio8 Realistic 0.6B](docs/models/synthesis/audio8.md) | [Apache-2.0](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b) |
| [Ming 3.0 TTS 0.5B](docs/models/synthesis/ming_tts.md) | [MIT](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B) |
| [Phonon 31M](docs/models/synthesis/phonon.md) | [Apache-2.0](https://huggingface.co/gradium/phonon-31m-en) |
| [OmniVoice](docs/models/synthesis/omnivoice.md) | [CC-BY-NC](https://huggingface.co/k2-fsa/OmniVoice#license) |
| [Kokoro 82M](docs/models/synthesis/kokoro.md) | [Apache-2.0](https://huggingface.co/hexgrad/Kokoro-82M) |
| [KittenTTS 2](docs/models/synthesis/kitten.md) | [Stellon Labs Community](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md) |
| [PersonaPlex 7B](docs/models/conversation/personaplex.md) | [NVIDIA Open Model](https://huggingface.co/nvidia/personaplex-7b-v1) |
| [Ming 3.0 Audio Tiny](docs/models/conversation/ming_audio.md) | [MIT](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny) |

Code, codecs, and other components can have separate terms.
