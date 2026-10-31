# Model definitions

Read docs/development/architecture.md and the model guide under docs/models/.

- Own identity, immutable checkpoint revisions, requests, and model constraints.
- Add a module per model, not an installable package.
- Import core when needed. Do not import runtimes or applications.
