# Test a Nemotron tool response

This experiment supplies a local JSON result to a tool request made by Nemotron.
It does not execute a tool. It uses NVIDIA's two-pass offline workflow.
Live microphone sessions do not accept tool responses.

## Use the website

1. Select Nemotron in the shared website.
2. Select **Offline tool-response test**.
3. Select **Record audio**, speak the request, then select **Stop recording**. You can also upload mono 16 kHz PCM16 WAV.
4. Enter tool definitions and a supplied response. The form includes a random-number example.
5. Start the test. Read the function request and the continued answer.

Recording requires trusted HTTPS and stops after 30 seconds. The uploaded input has the same duration limit.
Trailing silence gives the model time to
finish a request and an answer. The default is 16 seconds; the limit is 30 seconds.
The model must request the tool named in the supplied response. Otherwise, the test fails.
The workflow accepts one tool call and one supplied result per experiment.
A completed result means that EOTR and continuation text were observed. The fixed
audio timeline can still end before a longer spoken answer finishes.

## Use the CLI

Save tool definitions in `tools.json`:

```json
[
  {
    "type": "function",
    "function": {
      "name": "generate_random_number",
      "description": "Generate a random number between the requested minimum and maximum.",
      "parameters": {
        "type": "object",
        "properties": {
          "minimum": {"type": "integer"},
          "maximum": {"type": "integer"}
        },
        "required": ["minimum", "maximum"]
      }
    }
  }
]
```

Save a supplied result in `response.json`:

```json
{"tool_name": "generate_random_number", "response": {"result": 73}}
```

Record a request such as “Please use the random number tool to generate a random
number between one and one hundred.” Then run:

```sh
./voicelab run nemotron --input question.wav \
  --prompt 'Use the provided tool when asked for a random number. Read its result to the user.' \
  --trailing-silence 16 --tools-json tools.json --api-response-json response.json
```

The report records the detected call, response interval, predicted end-of-response
marker (EOTR), continued text, and timings. It hashes the supplied tool settings.
A failed or incomplete experiment retains its report and any completed audio.
Use ASCII tool definitions and results. JSON numbers must be finite.

## Interpret the result

The first pass predicts the call. The second pass inserts three segments at the
call's original audio frame: space for a regenerated call, supplied response tokens,
and one position for EOTR. The model predicts the call and EOTR; only response tokens
are forced. The inserted user-audio channel uses encoded silence.

The WAV retains the expanded timeline, including the inserted interval. It is not
edited into a seamless conversation. Input RTF counts both passes, preprocessing,
codec work, and output serialization. Its denominator is the original input plus
trailing silence. Loading is separate. This does not measure live tool latency.

A local acceptance test requested a random number, received `73`, predicted EOTR,
and continued with an answer containing seventy-three. CLI and shared FastAPI tests
passed. This establishes that roundtrip, not general tool accuracy, live tool support,
or numerical parity with NVIDIA's CUDA runtime.

## Source and implementation

The lab-owned decoder composes MLX-VLM 0.7.11 model primitives. NVIDIA's
[offline example](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/nemo/collections/speechlm2/inference/utils/offline_voicechat.py)
and [function-channel scheduler](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/nemo/collections/speechlm2/models/duplex_stt_model.py)
define the two-pass protocol. The model predicts function tokens greedily.

Definitions are in `voicelab_models.nemotron.tools`. Runtime preparation, decoding,
and response handling are separate modules under `voicelab_inference.nemotron.mlx`.
The runtime retains MIT and Apache-2.0 source licenses and an adaptation notice.
The checkpoint retains its OpenMDW-1.1 model license.
