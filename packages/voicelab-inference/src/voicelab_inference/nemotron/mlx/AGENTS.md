# Nemotron MLX

Read docs/models/conversation/nemotron.md and docs/usage/measurements.md.

- Give each model and stream an explicit owner.
- Synchronize device work before recording completed timings.
- Validate shapes, sample rates, finite audio, and cancellation.
- Give custom implementations distinct identities from the reference runtime.
- Do not infer ANE support from GPU execution or silently fall back to CPU.
