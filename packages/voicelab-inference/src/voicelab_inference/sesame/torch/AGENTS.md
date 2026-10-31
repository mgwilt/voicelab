# Sesame PyTorch

Read docs/models/synthesis/sesame.md.

- Preserve LICENSE and NOTICE. Record any change to source algorithm behavior.
- Serialize model access. Keep Mimi state local to each request.
- Restore patched forward methods on every exit.
- Validate vendor tensors with concrete types.
- Run `make check-sesame` during runtime changes, then `make check`.
