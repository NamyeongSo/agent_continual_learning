# Research code preservation · 2026-11-27

This branch preserves the Exgentic research workspace used for ACE and related
benchmark experiments. The original workspace is the Exgentic component under
Sico; the source base commit is recorded in `source_manifest.json`. The files
include local, uncommitted research development beyond that base commit.

- `src/`, `scripts/`, `tests/`, `docs/`, and configuration at the repository root
  contain the current working source.
- `run_code/` preserves Python/shell analysis scripts and serving templates
  previously stored under local experiment output folders. Paths mirror the
  original paths **below** `outputs/`. Historical `before` copies remain labeled
  as such; the reusable scripts at `scripts/ace/` are the current entry points.
- `run_settings/` contains a selected set of small sampling, budget, history,
  queue, and policy-hash settings. It contains no trajectories, results, learned
  playbooks, credentials, or model weights.
- `source_manifest.json` records original paths, byte sizes and SHA-256 hashes.

Raw experiment outputs, `team_prompt` histories/archives, benchmark installations,
virtual environments, caches, and private connection details are excluded. The
code is small enough for normal Git; this branch does not introduce Git LFS.

Some preserved analysis scripts use the original absolute workspace paths or
import helper modules from the original output tree. Those are historical code
snapshots, not self-contained result packages. To rebuild those figures, provide
the original data separately and restore/adjust the documented paths. Raw data
is not recreated by cloning this repository.

Historical analysis snapshots are checked with Python 3.12 syntax, matching the
environment in which they ran; some use Python 3.12 f-string syntax. The current
framework keeps its declared Python 3.11+ requirement.

Current reusable ACE entry points include:

- `scripts/ace/run_context_proposal_bottleneck.py`
- `scripts/ace/run_reflection_accumulation.py`
- `scripts/ace/run_advantage_accumulation.py`
- `scripts/ace/run_gemma4_train_only.py`
- `scripts/ace/rerun_gemma_format_errors.py`
- `scripts/ace/update_gemma26_diagnostic_figures.py`
- `scripts/ace/plot_independent_qwen_gemma_bars.py`
- `scripts/ace/plot_grouped_accumulated_independent.py`

The Gemma selected-case reruns and their figure updates are already complete.
The serving endpoint was stopped at the user's request. Cloning this branch
does not schedule experiments or restart a model server.
