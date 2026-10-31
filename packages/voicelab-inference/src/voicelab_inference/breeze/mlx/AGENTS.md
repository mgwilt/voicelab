# Breeze MLX

Read docs/models/synthesis/breeze.md.

- Preserve LICENSE and NOTICE.
- Keep vendor model details in vendor.py.
- Serialize model access. Reset the codec on every exit.
- Report a frame limit without EOS as incomplete.
- Run `make check-breeze` during runtime changes, then `make check`.
