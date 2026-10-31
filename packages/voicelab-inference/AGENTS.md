# Inference

Read docs/development/architecture.md and docs/usage/measurements.md.

- Use `voicelab_inference/<model>/<runtime>/`.
- Own loading, vendor adaptation, decoding, and device resources here.
- Import only this model's definitions, core, and declared runtime dependencies.
- Keep heavy imports lazy at application boundaries.
- Read the nearest runtime AGENTS.md before a runtime change.
