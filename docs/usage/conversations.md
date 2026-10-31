# Have a conversation

Start the lab:

```sh
./voicelab
```

1. Open **Conversation** and select a voice model.
2. Open **Voice and conversation settings** to change its controls.
3. Select **Start conversation**. Allow microphone access.
4. Wait for **Listening**, then speak.
5. Use **Mute**, **Interrupt**, or **End call** as needed.

Use trusted HTTPS from a phone. Read [network access](network.md).
Use headphones to reduce feedback. Calls last up to two minutes after model loading.
End the current call before another device starts one.

## Choose a conversation mode

- **Local voice cascade:** Whisper transcribes your speech. Qwen generates a reply. The selected TTS model speaks it.
- **Turn-based speech:** The selected speech-to-speech model receives each completed utterance directly.
- **Live duplex:** The model receives continuous microphone audio while it speaks. The model controls turn-taking.

The new managed cascade uses MLX on Apple Silicon.
The defaults are [Whisper large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) and [Qwen3-4B-Instruct-2507](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit) in 4-bit precision.
The first call downloads their pinned checkpoints. Inference stays local.

Cascade and turn-based modes use local speech and turn detection.
Each utterance must fit within 30 seconds.
The cascade completes its text reply before synthesis. Audio delivery still follows the selected voice model.
Buffered voice models remain buffered.

**Interrupt** clears queued speech in cascade and turn-based modes.
Some device operations must finish before the next response can start.
Native duplex models use their own interruption behavior; the separate interrupt control is disabled.

## Read the conversation

The view shows available model text and transcriptions.
Models without user transcription show **Audio message** for each input turn.
MiniCPM turn mode uses Whisper to supply its text-only history while processing the original audio itself.

Context is bounded and stays in memory for the call.
Interrupted answers do not enter later turn context.
A new call starts fresh. The app does not save recordings or transcripts automatically.

Timing labels distinguish input RTF, output RTF, and server first-audio latency.
They do not measure when sound reaches your speaker. Read [measurements](measurements.md).

## Use direct generation

Open **Direct run** to use the original model workflows, including tools, reference conditioning, and recorded inputs.
Switching workflows ends the current call. Model controls remain available in each workflow.

## Acceptance limits

Automated checks cover session ownership, signaling validation, audio transport, and context handling.
Physical mobile microphone, playback, and reconnect behavior require device testing.
Model execution does not establish intelligibility, speaker similarity, or parity with an upstream runtime.
