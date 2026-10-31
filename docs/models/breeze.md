# Breeze TTS

Breeze-TTS-2 generates streaming speech from text and a voice instruction.
The lab's runtime uses native MLX on Apple Silicon.
Output is mono 24 kHz PCM audio.

## Use the model

```sh
./voicelab run breeze \
  --text 'Hello. Welcome to Voice Lab.' \
  --instruction 'A calm, clear conversational voice.' \
  --output artifacts/runs/breeze-example --warmup --trials 3
```

For a browser test, select Breeze in the [shared website](../usage/README.md).
The runtime accepts text, a voice instruction, a seed, a guidance scale, and a frame limit.
Classifier-free guidance (CFG) controls the effect of the voice instruction through `--cfg-scale`.
First use downloads the pinned converted checkpoint, approximately 7.6 GB.

This runtime does not expose CUDA, TPU, or reference-audio conditioning.

## Measure performance

Output RTF divides generation time by emitted audio duration.
Timing excludes model loading.
It includes prompt encoding, sampling, codec work, device synchronization, and the output callback.
First PCM measures when the runtime produces decoded audio, before transport and playback.
Reports retain codec time and individual chunk arrivals.

Each regular chunk contains three codec frames, or 240 ms of audio.
A frame limit returns partial audio with `eos=false`.
The CLI marks that run incomplete and exits unsuccessfully.
Empty output is an error.
Codec state resets after completion, errors, and callback cancellation.
The shared website applies exclusive inference ownership and backpressure.
Read the [shared measurement rules](../usage/measurements.md) before comparing results.

## Implementation and evidence

Definitions live under `voicelab_models.breeze`.
The runtime lives under `voicelab_inference.breeze.mlx`.
`vendor.py` isolates MLX-Audio model and codec details.
The lab owns sampling, stream lifetime, and measurement policy.
The runtime uses MLX-Audio 0.6.3 and the Qwen3-TTS codec.
Language-model weights use bfloat16. Codec weights use float32.

Local acceptance includes a short instruction prompt, warmup, and three completed trials.
Real website tests also cover partial output, cancellation, and recovery.
This evidence does not establish broad language coverage, perceptual quality, voice cloning, or long-session reliability.
Keep device details and raw measurements in ignored artifacts.

## Sources and licenses

Sampling algorithms and failure-handling lessons come from [breeze-tts-mps](https://github.com/mgwilt/breeze-tts-mps), revision
`1367ec396248880496d5cfed213ef43b48d9640b`.
Keep the Apache-2.0 [LICENSE](../../packages/voicelab-inference/src/voicelab_inference/breeze/mlx/LICENSE)
and [NOTICE](../../packages/voicelab-inference/src/voicelab_inference/breeze/mlx/NOTICE).
The lab does not import that project's application, environment, benchmarks, or hybrid Torch/MLX runtime.

Source model: `BreezeBlue/Breeze-TTS-2`, revision
`799624c0b4a1daa8db6d28bbd9850043c0270734`.
Converted checkpoint: `mlx-community/Breeze-TTS-2-mlx`, revision
`3c8829fb7fd335818f085cd2ef49b4100c0e46c8`.

The converted checkpoint has not established numerical or audio-quality parity with the earlier hybrid MPS/int8 experiment.
Weights retain their [separate model license](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE).
The repository does not distribute model weights.
