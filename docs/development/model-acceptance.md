# Accept a model integration

Do not call a model complete because it loads, generates one clip, or appears in the catalog.
An adapter must preserve the released model's core inference workflows.

## Establish scope from the source

1. Read the official model card and the pinned reference implementation.
2. List released inference modes, conditioning inputs, languages, voices, and generation controls.
3. Separate hosted-only features from capabilities in the released checkpoint.
4. Record which runtime and interfaces implement each mode in the model guide.
5. Identify omissions explicitly. Do not describe an adapter omission as a model limitation.

Training, deployment, and accelerator optimization are separate work unless requested.
Do not create a placeholder runtime or silently select another device.

## Verify behavior

- Exercise every claimed mode with real weights, including reference conditioning when available.
- Check a changed reference or instruction reaches the model and changes generation with other inputs fixed.
- Test normal completion, a generation cap, cancellation, failure, and reuse.
- Verify audio format, finite samples, and input/output timing boundaries.
- Test the CLI and website request paths. A shared runtime test does not verify both interfaces.
- Compare custom decoding and conversions with the reference implementation before claiming parity.
- Evaluate intelligibility and speaker similarity separately from successful audio generation.

Retain scripts, inputs, reports, and audio in ignored artifacts.
Record public conclusions and unresolved checks in the model guide.

## State support precisely

Distinguish implemented, exercised with real weights, and evaluated for quality or parity.
Name the tested runtime and device class. Do not generalize from one device to all platforms.
Describe unavailable hardware and physical-browser testing as unverified acceptance checks.
The README matrix describes lab capabilities. It must not imply that unchecked features are absent upstream.
