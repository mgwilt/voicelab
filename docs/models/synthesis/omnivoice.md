# OmniVoice

OmniVoice generates multilingual speech through diffusion.
It supports automatic voices, voice design through instructions, and reference-based voice cloning.
Voice Lab uses the local OmniVoice 0.2.1 API and returns mono 24 kHz audio.

## Generate or design a voice

```sh
./voicelab run omnivoice --runtime torch --device mps \
  --text 'Welcome to Voice Lab.' --language en \
  --instruction 'A calm, clear voice.'
```

Omit `--instruction` to let the model choose a voice.
Use `--language auto` for language-agnostic generation. In the website, leave the language blank.
Supported devices are `cpu`, `mps`, and `cuda`; unavailable devices fail explicitly.
On MPS, the native audio tokenizer runs on CPU because its MPS operation is unsupported.
First use downloads approximately 3.3 GB of pinned model and tokenizer files.

## Clone a voice

```sh
./voicelab run omnivoice --runtime torch --device mps \
  --text 'Welcome to Voice Lab.' --reference-audio reference.wav \
  --reference-text 'The words in this recording.'
```

Use mono PCM16 WAV, 8–48 kHz, at most 30 seconds.
Cloning requires a transcript. The adapter does not load an automatic transcription model.
Choose a recording or voice-design instructions. The request rejects conflicting modes.
The website has reference-file and transcript controls.
Temporary files are removed after the request. Reports store recording hashes and metadata.

## Adjust generation

Run `./voicelab run omnivoice --help` for all options.
The website groups detailed controls under **Advanced generation**.

- `--steps` controls diffusion iterations; `--guidance-scale` controls classifier-free guidance.
- `--duration` sets the estimated output duration and takes precedence over `--speed`.
- `--t-shift`, `--layer-penalty-factor`, `--position-temperature`, and `--class-temperature` control iterative sampling.
- `--audio-chunk-duration` and `--audio-chunk-threshold` control native long-text chunking.
- `--denoise`, `--preprocess-prompt`, and `--postprocess-output` select native preprocessing and output processing.
- `--pad-duration` and `--fade-duration` control output edges. Postprocessing can change final waveform duration.
- `--seed` controls sampling.

Text normalization defaults to disabled.
English and Chinese normalization requires optional WeTextProcessing/Pynini dependencies, which are not installed by the lab.
Other languages require `num2words` for the native integer-normalization path.
A missing dependency causes an explicit error before model loading. The adapter does not silently ignore a missing normalization dependency.

## Read measurements

The native API returns a complete waveform. First PCM therefore measures full generation latency.
Real-time factor (RTF) includes reference conditioning, diffusion, decoding, and synchronization.
Loading and output callback time are excluded. Codec-only time and frame counts remain null.
Successful diffusion sets `completion=complete` and `eos=null`; no autoregressive EOS token exists.
Cancellation can stop delivery after generation. It cannot interrupt the buffered native call.

Local MPS runs exercised automatic generation and reference conditioning with actual audio.
The reference run changed fixed-seed PCM. Voice similarity, multilingual quality, and general performance remain unmeasured.
CPU and CUDA execution remain unverified. No MLX, TPU, or ANE adapter is included.

## Provenance

Model: [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice), revision
`c5fdb5ccb189668d56333f77ba2629f4cd7535f4`, including the audio tokenizer.
Runtime: `omnivoice==0.2.1`.
API source: [08be0b4ccbac3e13e374e86fbfead4b4cac343e2](https://github.com/k2-fsa/OmniVoice/tree/08be0b4ccbac3e13e374e86fbfead4b4cac343e2).
Code uses Apache-2.0. The upstream model card states CC-BY-NC terms for weights.
The lab does not distribute weights.
