# Write maintainable code

Use small modules with clear responsibilities. Split a module when unrelated responsibilities need different changes.
Preserve unrelated work.

## Choose functions and objects

Use objects for loaded models, sessions, mutable state, and resource lifetimes.
Use functions for parsing, validation, hashing, and audio transformations that do not retain state.
Prefer composition and narrow `Protocol` interfaces to deep inheritance.
Pass dependencies explicitly. Use context managers for resources that need closing.
Do not create a class only to hold static functions.

Follow the [dependency rules](architecture.md).
Do not add speculative engines, plugin frameworks, global registries, or generic `utils` folders.
Add a new backend contract when a concrete implementation requires it.

## Keep types precise

Annotate functions and meaningful object fields. Ruff checks annotations; ty treats all configured diagnostics as errors.
Use precise unions, enums, protocols, and immutable values.
Do not hide a mismatch with `Any`, unchecked casts, blanket ignores, or domain data stored in untyped dictionaries.

Adapt incomplete vendor APIs at one narrow boundary. Validate dynamically returned values there.
A necessary suppression must name the exact rule, reason, and supporting evidence.

Use frozen dataclasses for trusted internal values.
Use strict Pydantic models for structured external input, network messages, and stored schemas that can change.
Define how each boundary handles extra fields.
Validate input before model loading or expensive resource allocation.
Do not add Pydantic where no external validation boundary exists.

Reject invalid ranges, non-finite values, wrong shapes, and incompatible sample rates.
Give optional values a clear meaning.
Do not silently convert invalid data, change precision, or select another device.

## Keep runtime state local

Give each loaded model an explicit owner. Keep mutable session caches out of global state.
Reset or release resources after success, cancellation, and failure.
Restore temporary vendor changes on every exit.
Separate runtime mechanics from application policy.

## Check external APIs

For library or API usage, fetch current documentation with Context7 outside the sandbox.

1. Resolve the library with `npx ctx7@latest library <name> "<question>"`.
2. Read documentation with `npx ctx7@latest docs <returned-id> "<question>"`.
3. Prefer official sources.

Use at most three commands per question. Report quota failures.
Skip this lookup for ordinary refactors, business logic, and general programming.

## Finish the change

Follow [verification](verification.md). Update the document that owns the changed behavior.
Keep personal devices, paths, addresses, TLS files, preferences, and raw evidence in ignored `.local/` or `artifacts/`.
Use generic examples in public files.
