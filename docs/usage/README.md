# Use Voice Lab

Start the lab from the repository root:

```sh
./voicelab
```

Open `http://127.0.0.1:8443`. Select a model. Use the controls for that session.
A local configuration can change the address or enable HTTPS.

The launcher prepares dependencies and website files. The runtime loads model weights when you first use that model.
Only one website session can use inference at a time. Stop the current session before you start another.
The server stays available between sessions.

## Start a conversation

Select a model in **Conversation**, then select **Start conversation**.
Read [conversation usage](conversations.md) for modes, controls, and limits.

## Start a direct run

Open **Direct run**.

1. Open **Models**. Search by name, workflow, or runtime, then select a model.
2. Use **Controls** to supply inputs and adjust the model settings.
3. Select **Start session** or **Generate speech**. The first session loads the model.
4. Open **Output** to read measurements and available transcripts. The RTF readout also opens this view.
5. Use **Stop** to end the session. Read the model guide for drain and cancellation behavior.

Switching between **Controls** and **Output** preserves inputs and the active session.
Selecting another model cancels the current session and replaces its controls.

## Choose a task

- Save a model run: [CLI commands](cli.md).
- Connect a phone or another computer: [Network access](network.md).
- Read RTF and latency results: [Measurements](measurements.md).
- Set model controls: [Model guides](../models/README.md).

For prerequisites and supported models, read the [repository README](../../README.md).
