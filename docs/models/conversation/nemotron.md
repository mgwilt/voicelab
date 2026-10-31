# Nemotron VoiceChat

For ongoing calls, use [Conversation](../../usage/conversations.md). The direct workflows below remain available in **Direct run**.

NemotronLabs VoiceChat 11B supports duplex voice conversation and offline audio runs.
The lab uses a community MLX runtime on Apple Silicon.
Input is mono 16 kHz PCM16 audio. Output is mono 22.05 kHz PCM16 audio.

## Generation defaults

The runtime uses the checkpoint's decoding settings without application sampling overrides.
NVIDIA's [speech configuration](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B/blob/main/config.json)
sets guidance `0.2`, noise scale `0.001`, and top-p `0.95`.
The converted checkpoint carries those values into MLX-VLM.
The website exposes the seed and system prompt; generic temperature controls do not apply to every decoder here.

## Use the model

For a live test, start the [shared website](../../usage/README.md).
Select Nemotron. Allow microphone access. Use headphones and select **Start conversation**.

Stop ends input and drains queued model frames. A disconnect cancels the session.
Each session accepts at most 120 seconds of input.
The runtime receives continuous audio while it produces output. Voice activity detection (VAD) does not gate that timeline.
Nemotron sends cumulative user and assistant transcripts. The conversation view replaces each partial transcript as it grows or is revised.

For an offline test, run:

```sh
./voicelab run nemotron --input speech.wav
```

The input must be a mono 16 kHz PCM16 WAV.
The offline implementation uses upstream full-history generation and the built-in Aria voice.
It retains raw function-channel token IDs. The website displays function-channel text separately.
For a supplied tool response, use the website's **Offline tool-response test** workflow
or the [tool-response guide](nemotron-tools.md). This path uses two offline passes.
Live sessions display requests but do not accept tool responses. The lab does not execute tools.

## Measure streaming performance

Processing RTF uses the input timeline, including silence.
It divides synchronized processing time by input duration.
Read the [shared measurement rules](../../usage/measurements.md) before a comparison.

- Frame p50, p95, and maximum values measure processing time for native 80 ms frames.
- Frame processing includes model work, synchronization, and output conversion. Upstream profiling is enabled.
- An over-budget frame takes more than 80 ms to process.
- Queue age measures delay outside inference.
- A full queue of 25 frames stops the session. That queue represents 2 seconds of input.
- Setup time includes model loading, when needed, and new session initialization.

Processing RTF retains cold frames. It excludes session setup, network delay, and browser playback.
An RTF below 1 keeps up on average. An RTF above 1 accumulates lag.
These measurements do not prove turn-taking latency or long-session stability.

## Replay a measured input

Start the website first. Use a 16 kHz mono PCM16 WAV and a new output directory:

```sh
uv run --all-packages --extra mlx --locked python -m voicelab_web.models.nemotron.benchmark \
  --url wss://localhost:8443/api/nemotron/session --ca .local/certs/rootCA.pem \
  --input artifacts/fixtures/input.wav \
  --output artifacts/runs/streaming-trial --trials 3
```

This example uses the [local HTTPS configuration](../../usage/network.md).
The runner sends paced 80 ms chunks and adds five seconds of silence.
Each trial starts a new session. Later trials reuse the loaded model.
Reports retain checkpoint, source, runtime, and timing evidence. The runner also saves generated WAV files.
A replay does not test a mobile microphone.

## Implementation and evidence

### Known live-input limitation

The MLX streaming runtime can generate an answer from non-speech input.
A controlled test with quiet synthetic noise produced an unrelated answer without a user transcript or speaker playback.
It also transcribed and answered an attenuated recording of its own voice.
These observations do not establish the cause of every unwanted reply or the behavior of NVIDIA's reference runtime.
Browser echo cancellation does not resolve responses triggered by background noise alone.
Use **Output → Mute speaker** to test microphone input without assistant playback. This is a diagnostic control, not a model fix.

### Runtime and validation

The runtime lives under `voicelab_inference.nemotron.mlx`.
Definitions live under `voicelab_models.nemotron`.
The checkpoint is `mlx-community/NemotronLabs-VoiceChat-11B-8bit` at revision
`446fc83f6cde720e7fd9ad7e29d7acb3adbb8967`.
The runtime stores downloaded weights in the Hugging Face cache.

Local acceptance includes offline output and real WebSocket input/output through the shared server.
The supplied-response test detected a tool call, injected a known value, and produced
EOTR followed by an answer with that value through both CLI and FastAPI.
Browser playback and mobile microphone acceptance remain separate checks.
Upstream latency and ranking claims are not lab measurements.
This implementation is not NVIDIA's CUDA runtime.

## Sources and licenses

The source model is `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`, under OpenMDW-1.1.
MLX-VLM code uses the MIT license. Converted weights retain the model license.
Source references below were reviewed on 2026-10-30.

- [NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B).
- [MLX-VLM implementation](https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/nemotron_voicechat).
- [MLX-VLM API](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_voicechat/README.md).
- [Converted checkpoint](https://huggingface.co/mlx-community/NemotronLabs-VoiceChat-11B-8bit).

At that review, NVIDIA's reference setup used Linux/NVIDIA GPUs and Python 3.12 for its offline recipe.
Reconcile dependencies with the shared workspace before adding that runtime.
The [voicechat-mlx](https://github.com/huckiyang/voicechat-mlx) README listed missing encoder and duplex components at that review.
The lab therefore did not select it.
[speech-swift](https://github.com/soniqo/speech-swift/blob/main/docs/inference/voicechat.md) remains a comparison candidate with a different runtime and weight format.
