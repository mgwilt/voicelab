# CLI and runs

Read docs/development/cli.md and docs/development/coding.md.

- Construct runtimes here. Keep inference code in its runtime module.
- Separate argument parsing, run policy, and reports.
- Validate inputs before loading weights.
- Preserve failed reports and partial output. Never overwrite a run.
