# Use Voice Lab

Start the lab from the repository root:

```sh
./voicelab
```

Open `http://127.0.0.1:8443`. Select a model. Use the controls for that experiment.
A local configuration can change the address or enable HTTPS.

The launcher prepares dependencies and website files. The runtime loads model weights when you first use that model.
Only one website session can use inference at a time. Stop the current test before you start another.
The server stays available between tests.

## Choose a task

- Run a saved experiment: [CLI commands](cli.md).
- Connect a phone or another computer: [Network access](network.md).
- Read RTF and latency results: [Measurements](measurements.md).
- Set model controls: [Model guides](../models/README.md).

For prerequisites and supported models, read the [repository README](../../README.md).
