# KittenTTS 2

KittenTTS 2 generates expressive speech with built-in or cloned voices.
Voice Lab uses local KittenML 0.9.7 and emits mono 24 kHz audio.

## Generate speech

```sh
./voicelab run kitten --runtime torch --device mps \
  --text 'Welcome to Voice Lab.' --voice Bella --preset stable
```

Supported devices are `cpu`, `mps`, and `cuda`; unavailable devices fail explicitly.
Without a named voice or recording, the lab uses Bella.
The `expressive` preset supports expression experiments.
Native text syntax includes leading `[emotion]` tags, inline `<event>` tags, and `(((emphasis)))` spans.
First use downloads approximately 2.1 GB, including the separate speech decoder.

## Clone a voice

```sh
./voicelab run kitten --runtime torch --device mps \
  --text 'Welcome to Voice Lab.' --reference-audio reference.wav \
  --reference-text 'The words in this recording.' --use-reference-prompt
```

Use a clear recording of one speaker. Upstream recommends approximately 5–30 seconds.
The adapter accepts mono PCM16 WAV, 8–48 kHz, from 0.1 to 30 seconds.
Choose a named voice or a recording. Cloning requires a transcript; the adapter does not download an automatic transcription model.
The website exposes the same recording and transcript controls.
Temporary files and per-request cloned-reference caches are cleared after success, cancellation, or failure.
Reports store hashes and metadata instead of recording data.

## Adjust generation

Run `./voicelab run kitten --help` for all options.
The website groups detailed controls under **Advanced generation**.
Blank optional controls preserve native preset or checkpoint defaults.

- `--temperature`, `--top-k`, `--top-p`, and `--min-p` override sampling defaults.
- `--use-reference-prompt` includes reference codec tokens in the prompt; `--no-use-reference-prompt` uses speaker conditioning without that prefix.
- `--repetition-penalty`, `--repetition-window`, `--token-run-penalty`, and `--token-run-grace` control repetition.
- `--chunk-chars`, `--chunk-min-chars`, and `--max-tokens` control text splitting and per-sentence generation limits.
- `--use-emotion` or `--no-use-emotion` overrides automatic expression detection.
- `--seed` controls sampling. `--normalize` enables the native English-specific text normalizer.

## Read measurements

The native API emits sentence chunks. Short text normally produces one chunk after generation finishes.
First PCM measures the first decoded sentence. The adapter does not represent these as codec streaming frames.
Real-time factor (RTF) includes conditioning, generation, decoding, synchronization, and callback delivery.
Loading is excluded. Codec-only time remains null.

The adapter observes the resident language model's raw output before Kitten removes stop tokens.
A speech-end or stop token confirms natural completion. A sentence without either token marks the run incomplete.
Frame counts include generated speech codec tokens.
The observer is scoped to that resident instance and restored on every exit.
Cancellation closes the sentence iterator and restores random state at the next emitted chunk.
It cannot interrupt a blocking native sentence-generation operation.

Local MPS runs exercised built-in and cloned voices, confirmed natural stop tokens, and detected a forced token cap.
Conditioning changed fixed-seed PCM. These checks do not establish voice similarity or audio quality.
CPU and CUDA execution remain unverified. No MLX, TPU, or ANE adapter is included.

## Provenance

Model: [KittenML/kitten-tts-2](https://huggingface.co/KittenML/kitten-tts-2), revision
`c93cb7c271baa51d9ae8204d5277988906f235a2`, including speaker-encoder weights.
Runtime: `kittenml==0.9.7`.
Decoder: [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo), revision
`749d1c1a46eb10492095d68fbcf55691ccf137cd`, file `s3gen_meanflow.safetensors`.
A serialized, scoped vendor hook pins this decoder download because the native loader lacks a revision argument.
The language model expands packed ternary weights to BF16; the decoder uses float32.

Kitten weights retain the Stellon Labs Community License. The runtime package uses Apache-2.0.
The speaker model retains MIT terms. The decoder retains its upstream license and watermark behavior.
The lab does not distribute weights.
