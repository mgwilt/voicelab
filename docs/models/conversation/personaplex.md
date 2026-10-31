# PersonaPlex 7B

PersonaPlex generates speech and text while it receives speech. Voice Lab runs a custom MLX implementation locally on Apple Silicon.

## Run a conversation

Accept the [model license](https://huggingface.co/nvidia/personaplex-7b-v1). Authenticate with `hf auth login` if the download requires access.

Start the shared website with `./voicelab`. Select PersonaPlex and allow microphone access.

To replay a file, use mono PCM16 WAV at 24 kHz:

```sh
./voicelab run personaplex --input question.wav --voice NATF2.pt
```

Use `--prompt` to set the role. Use `--trailing-silence` to append silence, from 0 to 10 seconds.
The default tail is 3 seconds. Input and tail must fit within 120 seconds.

## Condition the voice

The `--voice` control selects a packaged NVIDIA voice. A reference recording replaces that selection:

```sh
./voicelab run personaplex --input question.wav --reference-audio speaker.wav
```

The reference must be mono PCM16 WAV at 8–48 kHz, from 0.4 to 30 seconds.
The runtime resamples it to 24 kHz and normalizes loudness to -24 LUFS, following NVIDIA's implementation.
It conditions assistant audio codebooks while supplying the fixed user-side sine tokens.
The browser accepts the same reference recording. Reports retain its hash, not its audio payload.

Use `--seed`, `--text-temperature`, `--audio-temperature`, `--text-top-k`, and `--audio-top-k` for sampling runs.
Temperature zero selects greedy decoding. The website groups these controls under Sampling controls.

## Read the measurements

Input real-time factor (RTF) divides synchronized model and codec processing time by input duration.
It excludes loading, prompt setup, file writes, and waits between frames.
A value below 1 means processing took less time than the supplied audio.

`first_pcm_seconds` measures cumulative processing until the first output frame. It does not measure conversational response latency.
`setup_seconds` includes loading and voice conditioning. The report retains the output WAV, text, and checkpoint revision.

PersonaPlex has no utterance completion signal. A stopped run means that the bounded capture ended.
It does not prove that an answer finished.

## Runtime and limits

The local scheduler supports all 16 depth codebooks, delayed streams, role text, and supplied voice prompts.
It uses MLX-Audio Moshi and Mimi primitives. It converts the original weights to eight-bit MLX weights in memory.
Initial loading therefore needs more memory than steady inference. No CUDA or TPU backend is implemented here.

The browser sends 80 ms PCM frames. The input queue holds at most two seconds.
A session fails if inference falls behind this bound. The file replay path runs without real-time pacing.

The implementation has scheduler and resource-lifetime tests. Full-weight local tests produced speech and text through the CLI and paced web-session transport.
Tests also covered raw-audio voice conditioning. Transcription recovered the emitted words.
The web test processed all supplied frames and closed without errors. It did not test a physical microphone or browser playback.
These short tests establish functional operation. They are not general performance, speaker-similarity, or audio-quality benchmarks.
The scheduler follows NVIDIA's delayed-stream and teacher-forcing rules. Numerical parity with NVIDIA CUDA inference has not been measured.

## Sources and licenses

- [NVIDIA model](https://huggingface.co/nvidia/personaplex-7b-v1/tree/fdaf4090a61cb315c138a1faee287ffd6c716309): NVIDIA Open Model License.
- [NVIDIA runtime](https://github.com/NVIDIA/personaplex/tree/3428dfd95309a7f3c84fd93259ded0f810d1ff91): MIT.
- [MLX scheduler reference](https://github.com/mu-hashmi/personaplex-mlx/tree/7b29cfc4d9110f6b9d831b141e62f3d16f0bbe77): MIT.
- [MLX-Audio](https://github.com/Blaizzy/mlx-audio): MIT; workspace version 0.6.3.

Source licenses and notices remain beside the adapter. See [measurements](../../usage/measurements.md) for comparison rules.
