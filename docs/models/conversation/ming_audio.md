# Ming Audio Tiny

Ming Audio Tiny processes speech and text. Its optional Talker produces spoken responses.
Voice Lab runs both components locally with MLX on Apple Silicon.

## Select a task

Start the shared website with `./voicelab`. Select Ming Audio and choose a task.
For audio input, select **Record audio**, speak, then select **Stop recording**.
Recording requires trusted HTTPS and stops after 30 seconds. You can also upload a mono 16 kHz PCM16 WAV of up to 30 seconds.
Select **Clear audio** to remove the input before a TTS task.

| Task | Input | Output |
| --- | --- | --- |
| `spoken_dialogue` | Audio, text, or both | Text and buffered speech |
| `audio_understanding` | Audio and instructions | Text |
| `asr` | Audio | Transcription and language label |
| `text` | Text, with optional audio | Text |
| `tts` | Exact text to speak | Speech |

```sh
./voicelab run ming-audio --input question.wav
./voicelab run ming-audio --task asr --input question.wav
./voicelab run ming-audio --task text --text 'Say hello.'
./voicelab run ming-audio --task tts --text 'Hello from Voice Lab.'
```

Use `--prompt` to change the instruction. `--text` is an alias.
For ASR, the default instruction asks for a language label and transcription.
Use `--max-tokens` to limit the text response, from 1 to 2048. The default is 256.

## Control speech generation

The website's speech generation section and CLI expose the same controls:

| Argument | Default | Range | Purpose |
| --- | --- | --- | --- |
| `--seed` | 0 | 0–4294967295 | Repeat the request's random sequence |
| `--max-decode-steps` | 375 | 1–1500 | Limit total Talker acoustic steps |
| `--cfg-scale` | 2 | 0–10 | Set conditional guidance |
| `--flow-steps` | 10 | 1–100 | Set flow solver steps per acoustic step |
| `--temperature` | 0 | 0–2 | Set stochastic solver noise |
| `--sigma` | 0.25 | 0–1 | Scale stochastic solver noise |

These controls apply to TTS and spoken dialogue. `--max-tokens` limits the Thinker;
it does not limit TTS speech. Temperature zero removes solver noise, but initial
latent sampling still depends on the seed. Sigma has no effect at temperature zero.
Each request restores the caller's random state after success, cancellation, or failure.
Reports retain all controls. A fixed seed supports comparisons within one runtime;
it does not guarantee identical results across hardware or library versions.

## Generate longer speech

The adapter uses native MLX-Audio punctuation and sentence segmentation. It packs
fragments up to 24 semantic units: one unit per English word or Chinese character.
Each fragment has its own decoder cache. Voice and reference conditioning apply
to every fragment. The native silence trim runs after each fragment, and the lab
concatenates the results before playback.

`--max-decode-steps` limits the whole request, unlike the native generator's
per-fragment limit. One acoustic step produces 80 ms before silence trimming.
The default 375 steps permits up to 30 seconds; 1500 permits up to 120 seconds.
If a fragment reaches the limit, or the budget ends before the next fragment,
the result is incomplete and retains all generated PCM. A complete result requires
end-of-sequence for every fragment. Cancellation checks run between fragments
and at each acoustic step.

## Supply a voice

Use `--voice` for a built-in voice. The default is `US_Stella`.
Use reference audio to condition the speaker instead:

```sh
./voicelab run ming-audio --task tts --text 'Welcome.' \
  --reference-audio speaker.wav
```

The reference must be mono PCM16 WAV at 8–48 kHz, from 0.4 to 30 seconds.
Add `--reference-text` with its transcript to enable acoustic continuation as well as speaker conditioning.
The website exposes both inputs.

Acoustic continuation trims trailing quiet samples and retains a 50 ms margin.
It uses a threshold of 1% of the reference peak, with a minimum amplitude of 0.0001.
Speaker extraction still uses the full reference. Use `--keep-reference-silence` to disable trimming for comparisons.
Long trailing silence can make continuation produce silence or unintelligible speech.

## Supply conversation context

Use `--system-prompt` for system instructions. Enable `--thinking` to retain a separate reasoning result.
The website has controls for previous user and assistant exchanges, including previous user audio.
For CLI runs, `--history history.json` accepts a JSON array of alternating turns:

```json
[{"role":"user","content":"My name is Alex."},
 {"role":"assistant","content":"Hello, Alex."}]
```

A turn can include `audio: {"wav_base64": "..."}` for a prior recording.
History permits 20 turns and 20,000 text characters. Combined audio must fit the 6 MiB request bound.
The final history turn must be an assistant answer. TTS accepts no conversation history.

## Read results

Speech output is mono PCM16 at 44.1 kHz. Responses are buffered before playback; this is not a duplex path.
Output real-time factor (RTF) divides inference and decoding time by output audio duration. Loading is separate.
Text-only tasks have no output-audio RTF or first-PCM measurement.

A completed response requires Thinker end-of-sequence, plus Talker end-of-sequence when speech is requested.
Incomplete reasoning is retained separately and is not spoken as an answer.
A token or duration cap preserves the report and partial output, then makes the CLI return an error.
Reports store reference hashes instead of recordings. See [measurements](../../usage/measurements.md).

## Verification and limits

Local full-weight tests covered all five tasks, reasoning, prior audio context, and both reference modes.
Transcription checks recovered the expected words from reference-conditioned speech.
A longer utterance spanning two native text fragments produced 12.9 seconds of
complete speech, identical to native MLX-Audio float32 output at the same seed.
Local ASR recovered the full text. A reduced aggregate step budget returned
partial speech with incomplete status.
These checks establish functional operation. They do not establish speaker similarity, general quality, or performance benchmarks.
Recording lifecycle tests use browser fakes. Mobile microphone capture and playback still require a test on the target device.

The pinned MLX Talker can omit words while reporting end-of-sequence. For example,
“Please generate a random number between one and one hundred.” lost words at
seeds 0 and 42 in both the lab and native MLX-Audio generation. Before native silence trimming, their PCM outputs
matched at each seed. Repeating the lab request after ASR produced identical PCM.
This reproduces a limitation in the current MLX path; it does not establish whether
the original CUDA implementation has the same issue. End-of-sequence proves that
generation stopped naturally, not that every requested word was spoken.
CUDA and TPU runtimes are not implemented. CPU tests cover validation, completion, and cancellation ownership.

The adapter corrects MLX-Audio's reference posterior to match the official Oobleck softplus scale.
It owns bounded decoding and preserves both completion signals.

## Sources and licenses

- [Original model](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny/tree/401b1abc63de0862dfbc9cd2a40895019d742d09): MIT.
- [MLX checkpoint](https://huggingface.co/mlx-community/Ming-3.0-audio-tiny-4bit/tree/af0238886184f15fb81e28858df2a3bf43983145): MIT.
- [Official Talker VAE](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny/blob/401b1abc63de0862dfbc9cd2a40895019d742d09/talker/AudioVAE/modeling_audio_vae.py).
- [MLX-Audio](https://github.com/Blaizzy/mlx-audio): MIT; workspace version 0.6.3.

The adapter retains its source license and notice beside the implementation.
