# Frontend

Read docs/development/frontend.md and docs/development/verification.md.

- Use vanilla TypeScript 7 components and Bun.
- Keep DOM code out of domain types.
- Decode unknown network data before use. Keep exhaustive union checks.
- Give audio, transport, and components explicit resource ownership.
- Test cancellation, late permission, playback completion, and stale callbacks.
