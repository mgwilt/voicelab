# Have a conversation

Start the lab:

```sh
./voicelab
```

1. Open **Conversation**. Use **Models** to select a voice model.
2. Open **Controls** to change the model and conversation settings.
3. Select **Start conversation**. Allow microphone access.
4. Wait for **Listening**, then speak.
5. Use **Mute microphone**, **Interrupt**, or **End call** as needed.

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
Qwen text uses the author's [recommended sampling settings](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507#best-practices):
temperature `0.7`, top-p `0.8`, top-k `20`, and min-p `0`. Replies remain capped at `256` tokens.
Each voice model uses the defaults recorded in its [model guide](../models/README.md).

Cascade and turn-based modes use local Silero VAD to detect speech.
Smart Turn v3 detects the end of each turn. Native duplex models control their own turn-taking.
Each utterance must fit within 30 seconds.
The cascade completes its text reply before synthesis. Audio delivery still follows the selected voice model.
Buffered voice models remain buffered.

**Interrupt** clears queued speech in cascade and turn-based modes.
Some device operations must finish before the next response can start.
Native duplex models use their own interruption behavior; the separate interrupt control is disabled.

## Check microphone processing

**Mute microphone** stops your voice from reaching the model. It does not silence the assistant.
To silence the assistant, open **Output** and select **Mute speaker** below the audio player.
The microphone, model, and transcripts continue running. Select **Unmute speaker** to hear the assistant again.
Speaker mute stays selected across calls until you change it or reload the page.
Use this control when phone volume buttons cannot silence call audio completely.

The page requests call audio routing before opening the microphone, when the browser supports this control.
It restores the previous audio mode after the call ends.
The browser requests echo cancellation, noise suppression, and automatic gain before connecting.
Open **Output**, then **Microphone processing**, to inspect the settings the browser reports.
An **On** value means the browser reports that processing is enabled. It does not measure how much echo it removes.
**Play and record** identifies the requested call audio mode. Browsers without this control manage routing automatically.
**Not reported** means the browser supplied no value; it does not mean the feature is off.
The page warns when echo cancellation is off, unreported, or processing changes fail.
These warnings remain visible when the call changes between listening and speaking.
If speaker audio still triggers replies, use headphones while checking the browser and device audio route.
The microphone stays open during assistant speech so interruption and duplex models continue to work.

## Read the conversation

The view shows available model text and transcriptions.
Open **Output** for transcripts and timing. Switching tabs preserves the call.
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
