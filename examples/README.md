# Component Examples

These scripts demonstrate individual agent behaviors with operational planning scenarios. They
are intentionally small API-backed smoke examples; deterministic orchestration coverage lives
in the repository's offline `tests/` suite.

Run an example from the repository root with module syntax:

```bash
uv run python -m examples.direct_prompt_agent
```

The available examples cover direct prompting, persona augmentation, bounded knowledge, RAG,
evaluation loops, semantic routing, and action planning.
