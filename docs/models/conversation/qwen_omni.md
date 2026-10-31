# Qwen3 Omni

Qwen3-Omni-30B-A3B-Instruct accepts speech or text and returns text with optional speech.
Voice Lab runs its Thinker, Talker, and speech codec locally through MLX on Apple Silicon.
The pinned 4-bit checkpoint requires approximately 22 GB of downloads, plus memory for inference.

## Run a voice conversation

Start the lab with `./voicelab`. Select **Qwen3 Omni**.
Record or upload a question. Select Ethan, Chelsie, or Aiden, then start the response.
You can also enter text without an audio recording.

For a retained experiment:

```sh
./voicelab run qwen-omni --input question.wav --prompt 'Answer briefly.' --output artifacts/runs/qwen-001
./voicelab run qwen-omni --prompt 'Say hello in Spanish.' --speaker Chelsie --output artifacts/runs/qwen-002
./voicelab run qwen-omni --help
```

Input files must contain mono PCM16 WAV audio, from 8 to 48 kHz, with at most 30 seconds per recording.
The processor resamples input to 16 kHz. Output uses mono PCM16 at 24 kHz.
Reference audio supplies conversation input. It does not clone the speaker's voice.

## Control a response

Use the prompt for questions, transcription instructions, language requests, or speaking instructions.
Instruction following can vary. A transcription request is not a separate forced-transcription decoder.
Keep the default system instructions unless your experiment requires a change.

The website exposes all controls under generation settings.
Use `--text-only` for text responses, or `--buffered` to deliver speech after decoding.
Use `--history` with a JSON file to provide previous complete user/assistant pairs.
Each item contains `role` and `content`. User items can also contain the shared `audio` object.
Assistant history contains text only. The website provides history fields without requiring JSON.

| Control | Purpose |
| --- | --- |
| `--speaker` | Ethan, Chelsie, or Aiden |
| `--seed` | Seed for Thinker and Talker sampling |
| `--max-tokens` | Thinker limit, from 1 to 2048 |
| `--temperature`, `--top-p`, `--top-k` | Thinker sampling |
| `--talker-max-tokens` | Speech frame limit, from 1 to 1500 |
| `--talker-temperature`, `--talker-top-p`, `--talker-top-k` | Talker sampling |
| `--talker-repetition-penalty` | Penalty on repeated speech tokens |
| `--chunk-frames` | Frames per PCM delivery, from 1 to 300 |

One speech frame represents 80 ms of output. The maximum output is 120 seconds.
The default delivery chunk contains 25 frames, or two seconds of audio.
Both decoders must emit their end token for a speech run to complete.
A generation cap retains partial output and reports an incomplete run.

## Understand timing and limits

This runtime uses recorded turns. It does not support full-duplex conversation or live interruption during input recording.
The Thinker finishes its text before the Talker starts speech.
Streamed PCM then arrives as the Talker produces codec frames.
The codec retains 25 preceding frames when decoding each new window.

Output real-time factor (RTF) divides response time by emitted audio duration.
Response time includes input processing, both decoders, codec work, and delivery backpressure.
Checkpoint loading is measured separately. First PCM measures time before the first audio callback.
Text-only responses have no output RTF or first-PCM value.

Cancellation stops between decode steps. Model loading cannot be interrupted inside the vendor loader.
The server retains its inference worker until loading or decoding stops and device cleanup completes.

The current voice workflow excludes image input, video input, structured tool execution, and voice cloning.
It uses the Instruct checkpoint, including the Talker and codec; it does not substitute a text-only Thinking checkpoint.

## Inspect provenance

- Original model: [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct), revision `26291f793822fb6be9555850f06dfe95f2d7e695`.
- Local checkpoint: [MLX 4-bit conversion](https://huggingface.co/mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit), revision `93b3cbddd65ed4babff8f22fba491cdba7a21778`.
- Model weights: Apache-2.0.
- Runtime primitives: `mlx-vlm==0.7.11`, MIT. Attribution and license are beside the runtime source.

Each CLI run retains `run.json`, `response.txt`, and speech in `response.wav`.
The report records checkpoint revisions, controls, completion evidence, timing boundaries, package versions, and source hashes.

Read [measurements](../../usage/measurements.md) before comparing runtimes.

## Verification status

Real-weight checks on Apple Silicon exercised speech input, text input, text-only output, all three voice presets, and text or audio history.
A request with both previous and current audio completed successfully.
Independent ASR recovered the Spanish greeting, with a small wording difference from the requested text.
The CLI and shared FastAPI WebSocket route both completed a spoken answer to a recorded question.

The audit checked decoder caps, cancellation after first PCM, and reuse after cancellation.
A short deterministic buffered response matched native mlx-vlm PCM exactly.
Native and lab Thinker text matched for repeated audio input.

A 12.56-second response produced equal sample counts with buffered delivery and chunks of 1 or 25 frames.
Independent Ming ASR recovered the same complete transcript from all three outputs.
Windowed streaming was not bit-identical to buffered decoding; measured mean squared error was below `5e-6` on normalized PCM.
These checks establish intelligibility for the tested clips, not a general quality ranking.

Physical mobile playback, NVIDIA execution, and TPU execution remain unverified for this runtime.
