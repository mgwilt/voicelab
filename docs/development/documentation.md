# Write repository documentation

Use a practical style based on [ASD-STE100](https://www.asd-ste100.org/), Issue 9.
Aim for close alignment, not a measured percentage or certified compliance.
Keep technical names and exact syntax when simpler words would change the meaning.
The repository does not claim a complete approved-dictionary audit.

## Write clear sentences

- Use active voice and direct commands.
- Give one instruction per sentence.
- Keep instructions within 20 words where practical.
- Keep descriptions within 25 words where practical.
- Put a condition before the action that depends on it.
- Use short paragraphs with one topic.
- Use the same term for the same concept.
- Define an abbreviation before its first substantive use on a page.
- State the actor, action, and expected result.

Preserve commands, paths, identifiers, versions, revisions, license names, and quoted source text.
Do not shorten a sentence by removing required articles or useful meaning.
Keep commands out of notes. Put procedures in their execution order.

## Reveal detail by task

Keep `AGENTS.md` and `docs/README.md` as short indexes. Use plain lists in agent indexes, not tables.
Keep the root README focused on purpose, supported workflows, and startup.
Tables can compare models or command options in user guides.

Use three documentation areas:

- `usage/`: tasks for people who run the lab.
- `development/`: ownership, implementation rules, and verification.
- `models/`: controls, runtime behavior, timing boundaries, and provenance for each model.

Give each fact one owning page. Link to that page instead of repeating its rules.
Keep model-specific settings out of shared usage pages.
Keep source directories' `AGENTS.md` files short and local.
Add a page only when it serves a distinct task or reference need.

## Use consistent terms

- **Model:** the learned voice system and its identity.
- **Runtime:** an implementation that executes a model.
- **Device:** the hardware target that a runtime uses.
- **Package:** an installable workspace distribution.
- **Module:** a Python source unit within a package.
- **Run:** one execution whose outputs and status are retained.
- **Session:** an interactive connection and the resources it owns.

Keep API names such as `SynthesisBackend` unchanged, even when prose uses a different general term.

## Review a rewrite

1. Map old requirements, commands, limits, and provenance to their new pages.
2. Check claims against source code and retained evidence.
3. Update all internal links and agent entry points.
4. Check sentence length and repeated rules.
5. Check that each page is reachable from its section index.
6. State any unverified claims or acceptance limits.

Sentence-length checks assist review. They do not prove ASD-STE100 dictionary or grammar compliance.
