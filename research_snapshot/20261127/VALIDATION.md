# Preservation validation

Validation ran against the new standalone checkout, with imports verified to
resolve to its `src/exgentic/`. The existing Python 3.12 environment from the
source workspace supplied dependencies; no model or benchmark inference was run.

- ACE, AppWorld scoring/budgets, GPU routing, accumulation and format-repair
  checks: **122 passed, 3 skipped**.
- Isolated runner suite, with `-p no:faulthandler`: **138 passed, 11 skipped**.
- Ruff syntax/undefined-name checks passed for current source with the project
  Python target; historical analysis snapshots passed with Python 3.12.
- All **978 Python source files** parsed successfully with Python 3.12.
- `uv lock --check --offline` passed.
- Standalone pre-commit Ruff, secret scan, and added-file size checks passed.
- Every indexed file is below **2 MiB**. The largest file is an existing
  documentation screenshot, approximately 1 MiB. No Git LFS was introduced.
- No raw experiment output, model weights, rollout history archives, virtual
  environments or private handoff files are indexed. Ignore patterns were
  checked against representative sensitive/generated paths.

The secret scan's reviewed matches are reproducibility hashes, local vLLM
`EMPTY` placeholders, descriptive run-directory names, and synthetic or unchanged
recorded-test fixtures. The audited baseline stores hashes, not credential values.

This is a source preservation branch, not a claim that every historical script
is portable or every integration was rerun. Optional checks skipped by the test
suite remain skipped; raw data and server-specific resources are not included.
