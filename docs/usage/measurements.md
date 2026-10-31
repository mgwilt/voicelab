# Read and compare results

A successful run proves only what that run measured. Separate speed, audio quality, numerical agreement, and device support.

## Read RTF

Real-time factor (RTF) is processing time divided by audio duration.

- Input RTF uses the incoming audio timeline. That timeline can include silence.
- Output RTF uses the duration of generated audio.
- An RTF below 1 means processing is faster than the measured audio timeline.

Do not compare input RTF with output RTF as if they measured the same work.
Read the model guide for the exact timing boundary.

## Read latency

First PCM measures when the runtime produces its first decoded audio chunk.
Frame latency measures processing time for a model frame.
Some live workflows report compute time to first PCM. This excludes waits between input frames.
Neither value includes all network, browser queue, and speaker delay.
Neither value alone proves conversational turn-taking latency.

Report model loading separately from generation. Synchronize device work before you record completed processing time.
State whether a result includes cold frames, prompt processing, codec work, or output delivery.

## Compare runs

1. Keep the input, prompt, seed, conditioning, and model lineage constant.
2. Record the runtime, device, precision, and quantization settings.
3. Run a warmup.
4. Repeat the measured trial.
5. Check audio quality when precision, codecs, or implementations change.

A short test does not prove long-session reliability. Output from one device does not prove support on another device.
MLX GPU execution does not prove Apple Neural Engine (ANE) support.

## Keep evidence

Create a fresh output directory for each run. Do not overwrite a run.
Retain these details when the workflow supplies them:

- Model and converted checkpoint identities, with immutable revisions.
- Runtime identity, settings, software versions, and hardware.
- Input, reference, and output hashes.
- Source and dependency-lock hashes.
- Auxiliary codec checkpoint revisions and installed text frontend package versions.
- Raw output, status, timing values, and measurement limits.

Label warmup reports separately. Keep failed reports and partial audio.
For synthesis, an end-of-sequence (EOS) event can confirm completion.
Models without EOS can confirm completion through their generation algorithm.
If an upstream API hides its stop reason, the report marks completion as `unverified`.
An unverified result is not a completed synthesis benchmark.
A frame limit without EOS means incomplete output. Do not count that result as a completed synthesis benchmark.

Keep raw evidence under ignored `artifacts/`. Review evidence before public release.
Keep personal device details out of public documentation.
