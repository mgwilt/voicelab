# Audio8

Audio8 Realistic 0.6B generates 44.1 kHz speech in 23 languages. Voice Lab runs
the released model and codec locally through MLX-Audio's ArkTTS architecture.

## Run

```sh
./voicelab run audio8 --text 'Hello from Voice Lab.'
./voicelab run audio8 --text 'A new sentence in the reference voice.' \
  --ref-audio reference.wav --ref-text 'Exact words in the reference clip.'
```

Reference audio and its transcript are required together. Use mono PCM16 WAV,
8–48 kHz, up to 30 seconds. The runtime resamples it to the codec's 44.1 kHz rate.
The website provides the same reference upload and transcript fields.
The model infers language from the text; there is no language-selection token.

Supported languages are Arabic, Cantonese, Chinese, Dutch, English, Finnish,
French, German, Hindi, Indonesian, Italian, Japanese, Korean, Malay, Norwegian,
Polish, Portuguese, Russian, Spanish, Swedish, Thai, Turkish, and Vietnamese.
Coverage follows the model card; quality varies by language.

Use `--temperature`, `--top-p`, `--top-k`, `--seed`, and `--max-frames` for sampling.
`--no-do-sample` selects greedy decoding; `--top-k 0` disables top-k filtering.

## Measurements and limits

The runtime generates codec frames, then decodes a complete waveform. First PCM
measures full-output availability. This adapter does not stream partial audio.
The effective context budget includes reference codec frames and both text parts.
A context or frame cap marks incomplete output; natural EOS marks completion.
Loading includes conversion of pinned PyTorch codec tensors into MLX arrays.
Generation includes reference encoding, prompt processing, decoding, and delivery.
Cancellation releases caches after the bounded generation finishes.

This runtime requires Apple Silicon macOS and uses float32 arithmetic. Listening
quality and numerical parity with the official CUDA path require separate evidence.
Local reference-conditioned and greedy smoke tests completed. Local ASR recovered
the requested new sentence from cloned output. The shared FastAPI reference-upload
route also returned complete PCM. This is content validation, not a voice-similarity score.
See [measurement rules](../../usage/measurements.md).

## Provenance

[Edge0/Audio8-TTS-Realistic-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b),
revision `1e857568bfd4f02931aca3988ee773a44d2d26d3` supplies both model and codec.
The loader reads tensor data and does not execute checkpoint Python files.
Weights retain Apache-2.0 terms.
