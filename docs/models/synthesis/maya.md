# Maya 2 Native

Open **Direct run** for the synthesis workflows below. Use [Conversation](../../usage/conversations.md) for ongoing voice calls.

Maya 2 Native generates multilingual speech with named or cloned voices.
Voice Lab runs the native MLX-Audio model locally and streams 24 kHz mono audio.

## Generate speech

Use Apple Silicon macOS. The first run downloads approximately 3.8 GB of model, codec, and speaker-encoder files.

```sh
./voicelab run maya --runtime mlx --text 'Welcome to Voice Lab.' \
  --language en --voice Siddhant
```

Language codes are `as`, `bn`, `en`, `gu`, `hi`, `kn`, `ml`, `mr`, `or`, `pa`, `ta`, and `te`.
The website lists 20 bundled voices. Unknown names fail validation before model loading.
Without a named voice or recording, the lab uses Siddhant.

## Clone a voice

```sh
./voicelab run maya --runtime mlx --text 'Welcome to Voice Lab.' \
  --reference-audio reference.wav --reference-text 'The words in this recording.' \
  --conditioning icl_spk --ref-language en
```

Use a mono PCM16 WAV file, 8–48 kHz, from 0.1 to 30 seconds.
Choose a named voice or a reference recording. The website selects reference conditioning when a recording is supplied.

- `icl_spk` uses reference codec tokens and a speaker embedding.
- `icl` uses reference codec tokens.
- `spk` uses only a speaker embedding. It does not require a transcript.

Both in-context modes require the reference transcript.
`--ref-language` identifies its language; the native default uses the target language.
Temporary files are removed after each request. Reports retain hashes and metadata instead of recording data.

## Adjust generation

Run `./voicelab run maya --help` for all options.
The website groups sampling controls under **Advanced generation**.

- Backbone sampling: `--temperature`, `--top-k`, `--top-p`.
- Depth sampling: `--depth-temperature`, `--depth-top-k`.
- Repetition: `--presence-penalty`, `--presence-window`.
- Limits and delivery: `--min-frames`, `--max-frames`, `--streaming-interval`.
- Text and randomness: `--normalize` or `--no-normalize`, and `--seed`.

## Read measurements

Real-time factor (RTF) includes conditioning, generation, decoding, synchronization, and callback delivery.
Loading is measured separately. Codec-only time remains null.
The native EOS probability guard marks completion. Reaching the frame limit marks incomplete output.
Cancellation closes the generator and restores random state at the next decoded chunk boundary.

Local runs exercised named voices, all three reference modes, natural completion, caps, cancellation, and reuse.
Reference modes changed fixed-seed PCM. Speaker similarity, multilingual quality, and cross-platform parity remain unmeasured.
CUDA, TPU, and ANE runtimes are not included.

## Provenance

Source: [maya-research/maya2-native](https://huggingface.co/maya-research/maya2-native), revision
`76aedafc325fb72413569a0e15121d58ea2ac312`.
Conversion: [aruntemme/maya2-native-mlx-8bit](https://huggingface.co/aruntemme/maya2-native-mlx-8bit), revision
`ee18411e5f4f69dcd526b66d2b18dbb8e2f21c8e`.
The backbone uses 8-bit weights. Other weights use BF16, with FP16 depth inference.
Runtime: `mlx-audio==0.6.3`, native `maya2` implementation.
Mimi: [kyutai/moshiko-pytorch-bf16](https://huggingface.co/kyutai/moshiko-pytorch-bf16), revision
`2bfc9ae6e89079a5cc7ed2a68436010d91a3d289`, file `tokenizer-e351c8d8-checkpoint125.safetensors`.
Speaker encoder: [aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX](https://huggingface.co/aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX), revision
`e749e6e08557f4a3ceb6ce3bf6f0b79efe592a77`.

Maya and the speaker encoder retain Apache-2.0 terms. MLX-Audio uses MIT terms. Mimi weights retain CC-BY-4.0 terms.
The lab does not distribute weights or claim numerical parity with the source runtime.
