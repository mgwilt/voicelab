# MiniCPM-o 4.5

For ongoing calls, use [Conversation](../../usage/conversations.md). The direct workflows below remain available in **Direct run**.

Run speech conversations with the local `llama.cpp-omni` runtime on Apple Silicon.
The lab supports live duplex sessions and recorded-audio or text questions with spoken answers.

## Start

```sh
./voicelab
```

Select **MiniCPM-o 4.5**. Choose **Live conversation** or **Spoken answer**.
Use trusted HTTPS for microphone access from another device.

The first run downloads about 8 GB and builds pinned native code.
Install Git, CMake, and a C++ compiler before that run.
The launcher owns the private inference process. No separate server is required.

For a recorded question:

```sh
./voicelab run minicpm --input question.wav --output artifacts/runs/minicpm-001
```

For a text question:

```sh
./voicelab run minicpm --text "What is the capital of France?" \
  --output artifacts/runs/minicpm-002
```

For a duplex replay:

```sh
./voicelab run minicpm --mode duplex --input question-16k.wav \
  --trailing-silence 5 --output artifacts/runs/minicpm-003
```

Use a new output directory for each run. Failed runs retain their report and partial audio.

Use `--help` for all CLI controls.
`--history` reads a JSON array of alternating `{"role":"user","text":"..."}` and assistant objects.

## Modes and controls

| Control | Behavior |
| --- | --- |
| Live conversation | Listen and speak concurrently; stop after at most two minutes. |
| Spoken answer | Submit a recording, text, or both. Output arrives as PCM chunks. |
| Instructions | Set the system prompt for either mode. |
| Previous conversation | Supply up to three text message-and-answer pairs for spoken answers. |
| Audio context reference | Condition the language/audio encoder with 1–30 seconds of reference audio. |
| Generation controls | Set seed, text temperature, top-p, top-k, speech temperature, and length penalty. |
| Live controls | Set listen probability scale and speech tokens per input chunk. |
| Response token limit | Bound a spoken answer; capped output is incomplete. |

The released voice workflows target English and Chinese. Other languages need separate evaluation.

Input recordings use mono PCM16 WAV. Spoken answers accept 8–48 kHz and up to 30 seconds.
Duplex replays require 16 kHz. The browser sends one-second frames at 16 kHz.
Output uses mono PCM16 at 24 kHz.

The native waveform decoder uses its fixed speaker prompt cache.
An audio context reference does **not** implement speaker cloning in this adapter.
Speaker similarity needs a separate evaluation.

Text history is supported. Audio-bearing history is not implemented in this native adapter.
Image and video workflows are outside the lab's voice interface.
The adapter does not expose upstream training or text-only response workflows.
CUDA, TPU, MLX, and neural-engine execution are not implemented by this adapter.

## Measurements and stopping

For spoken answers, real-time factor (RTF) is generation time divided by output audio duration.
For duplex, RTF is processing time divided by submitted input audio duration.
Both include native inference, waveform decoding, transport delivery, and final drain.
Neither includes model preparation or the wait between live input frames.

First PCM measures processing time until the first audio bytes arrive.
For duplex, this excludes microphone pacing. It is not conversational response latency.
A duplex capture stop is not an utterance end-of-sequence event.

Stop closes the microphone and drains submitted frames. Stop again to cancel immediately.
Disconnects cancel preparation or inference and reap the owned process before releasing the model lease.
The server rejects a backlog above four input frames.

## Runtime and source

- Source model: [openbmb/MiniCPM-o-4_5](https://huggingface.co/openbmb/MiniCPM-o-4_5), revision `eb4141de24550a3a1a4082f718be58f46b761728`.
- Native weights: [openbmb/MiniCPM-o-4_5-gguf](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf), revision `db25077c33951fe163b42986fba0132e279872a2`.
- Precision: `MiniCPM-o-4_5-Q4_K_M.gguf` language model; F16 audio, speech, projection, and waveform modules.
- Native source: [OpenBMB/llama.cpp-omni](https://github.com/OpenBMB/llama.cpp-omni/tree/a64932ec615e1463158129feaf2f0752d0c07a2e), revision `a64932ec615e1463158129feaf2f0752d0c07a2e`.
- Weight license: Apache-2.0. Native source license: MIT.

The native runtime uses Metal for model inference and CPU waveform vocoding.
The upstream initializer also loads its vision projection module, although this interface sends no images.
Local patches correct isolation, prompt handling, text delivery, acoustic truncation, and completion signaling.
See the [native notice](../../../packages/voicelab-inference/src/voicelab_inference/minicpm/native/NOTICE).
Reports record checkpoint revisions, precision, runtime, input hashes, and the patch hash.

## Acceptance limits

Real-weight checks passed for text answers, recorded speech answers, and duplex speech on Metal.
The normal CLI verified automatic source preparation. Website tests verified both turn and duplex request paths, PCM delivery, and transcripts.
Independent transcription recovered the full spoken answer: “the capital of france is paris.”

A one-token generation cap returned partial speech with `completed: false`.
Cancellation before startup and during generation passed. Reusing the runtime after completed turns also passed.
An uploaded reference reached the audio encoder; this does not establish speaker cloning or similarity.

One short spoken-question run measured output RTF **0.575** and first PCM **0.810 seconds**.
The corresponding website turn measured output RTF **0.576**.
A duplex replay measured input RTF **0.157** across 11 seconds of input, including silence.
These are single-run measurements, not a hardware ranking or conversational latency benchmark.

Physical mobile microphone/playback behavior, speaker similarity, and parity with upstream Python inference remain unverified.
