# Verify a change

Test observable behavior and invariants. Do not test only implementation details.

## Run checks

Run focused tests during implementation. Run the complete check before you finish:

```sh
make check
```

This command runs frontend checks, Python lint and type checks, integration tests, and runtime tests.
Use these targets for focused work:

```sh
make check-web
make check-sesame
make check-breeze
```

## Select evidence

Use small injected fakes for request validation, experiment control, and resource cleanup.
Keep root contract tests independent of weights, network access, microphones, and accelerators.
Keep runtime-specific tests with their runtime owner.
Keep shared integration and dependency tests under `tests/`.

Cover malformed inputs, failed runs, incomplete output, cancellation, and artifact preservation.
Make tests deterministic where possible.
A fake runtime does not prove inference. An import does not prove model execution.
Offline output does not prove streaming behavior or performance.

Run real model experiments separately. Retain their evidence under ignored `artifacts/`.
Use the [measurement procedure](../usage/measurements.md) for performance claims.

## Verify the website

Test wire decoding, model changes, stale callbacks, backpressure, and resource release.
Check late microphone permission, playback completion, and cancellation.
Keep the audio resampler's timeline continuous across input blocks.

A build or DOM test does not prove native Web Audio behavior.
Test mobile microphone access and playback on the actual browser and device.
Report unavailable hardware or browser access as an acceptance limit.
