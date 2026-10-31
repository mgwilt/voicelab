# Shared tests

Read docs/development/verification.md.

- Keep these tests independent of weights, network access, and accelerators.
- Use injected fakes for contracts and application behavior.
- Cover invalid input, failure, cancellation, and retained artifacts.
- Keep runtime-specific tests with their runtime owner.
