# Sesame CSM

CSM-1B generates mono 24 kHz speech from text and optional reference speech.
It is a contextual synthesis model, not the hosted Sesame assistant or a duplex conversation model.
Speaker IDs select conversation roles. They do not select named preset voices.

## Use the model

```sh
./voicelab run sesame \
  --device mps --text 'Hello from Voice Lab.' \
  --output artifacts/runs/sesame-first --warmup --trials 3
```

Select `cpu`, `mps`, or `cuda` explicitly. An unavailable device causes an error without a device fallback.
First use downloads a pinned, gated checkpoint.
Complete the access requirements on [Hugging Face](https://huggingface.co/sesame/csm-1b) if needed.
Authenticate the local Hugging Face client. Do not put access tokens in project files.

For a browser test, select Sesame in the [shared website](../../usage/README.md).
The page provides text, speaker, reference audio, transcript, sampling, seed, and frame-limit controls.
Reference conditioning is available through the website, Python, and the run command.

## Configure browser inference

Add these model settings to ignored `.local/server.toml`:

```toml
sesame_device = "mps" # cpu (default), mps, or cuda
sesame_engine = "direct-depth" # direct-depth (default) or baseline
```

Network and TLS settings are in the [network guide](../../usage/network.md).

## Supply reference speech

Add both options to the run command:

```sh
--reference-audio .local/reference.wav --reference-text 'The reference transcript.'
```

Use a matching transcript. Reference conditioning does not guarantee speaker similarity.
The CLI's single-reference loader accepts 24 kHz WAV and downmixes stereo input.
Website uploads and context manifests require mono PCM16 WAV at 8–48 kHz; the runtime resamples them to 24 kHz.
References must contain no more than 30 seconds of audio.
The reference, text, and requested frames must fit the 2048-position context.
The runtime checks the tokenized length before generation.

For multiple prior turns, use the website's conversation context controls or `--context context.json`.
The manifest contains an ordered list of `audio`, `text`, and `speaker` objects.
Audio paths are relative to the manifest. Speaker values are 0 or 1.
Supply at most four turns with a combined audio duration of at most 30 seconds.
Use either conversation context or a single reference pair.

## Measure performance

Output RTF divides generation time by emitted audio duration.
Timing starts after tokenization. It excludes model loading and includes codec decoding, consumer delivery, and backpressure.
`first_pcm_seconds` records the first decoded chunk ready for delivery.
Network delay, browser queues, and speaker playback add latency after that point.

The first chunk requires four 80 ms frames.
End-of-sequence (EOS) is required for completion.
A frame limit produces an incomplete report and partial audio.
Read the [shared measurement rules](../../usage/measurements.md) for comparisons and evidence retention.
Reports also include load timing, reference hashes, and separately labeled warmups.

## Implementation and evidence

The runtime lives under `voicelab_inference.sesame.torch`.
Definitions live under `voicelab_models.sesame`.
The shared workspace uses Python 3.14.8, PyTorch 2.14.1, and Transformers 5.22.1.
CPU inference uses float32. MPS and CUDA use bfloat16 with a float32 Mimi codec.

`direct-depth` uses integer-indexed depth heads and a frame-local key/value cache.
It retains all codebooks and reference sampling behavior.
`--engine baseline` uses Transformers depth generation for comparison.
Both engines use the same Mimi stream adapter.
Mimi retains causal convolution history, transpose-convolution overlap, and transformer cache state.
The runtime restores temporary forward-method changes on every exit.

The shared website bounds output queues. A disconnect cancels generation at a chunk boundary.
Device loading and active kernels must finish before another session can own inference.

MPS inference has run locally. CPU and CUDA paths still need separate hardware acceptance.
This model has no MLX, TPU, or ANE implementation in the lab.
Tiny-model tests cover sampled and greedy depth parity, causal convolution, EOS, cancellation, final flush, restoration, and transport failures.
A real website transport test produced completed audio using an uploaded reference and transcript.
Those tests do not establish speaker similarity, perceptual quality, full-checkpoint parity across devices, or mobile browser acceptance.

## Sources and licenses

The adaptation comes from [mgwilt/csm-mps](https://github.com/mgwilt/csm-mps), revision
`97f4e768f07e0572649b5937fa6ca1855c5239d1`.
Keep its Apache-2.0 [LICENSE](../../../packages/voicelab-inference/src/voicelab_inference/sesame/torch/LICENSE)
and [NOTICE](../../../packages/voicelab-inference/src/voicelab_inference/sesame/torch/NOTICE).
The separate model checkpoint is `sesame/csm-1b` at revision
`c92a71e1c419772e25be7dc14d952c2521a740ab`.

The adapter does not apply the upstream standalone generator's SilentCipher watermark.
Do not describe generated audio as having that watermark.
The implementation follows the [Transformers CSM documentation](https://huggingface.co/docs/transformers/model_doc/csm)
and the locked Transformers source.
