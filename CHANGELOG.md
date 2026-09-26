# Changelog

## Unreleased

### Added

- Terminal-Bench 2.0 adapter using pinned Harbor Docker environments and official verification.
- ACE training with Terminus-2 as the default Terminal-Bench actor, trajectory reflection, and playbook transfer between tasks.
- PREMiSE pipeline support for BFCL, AppWorld, and Terminal-Bench with frozen task splits.
- Five-task ACE pilot wrapper with a 15-turn limit and 8,192-token reflection output budget.

### Fixed

- Record subprocess startup and task execution errors per task, then continue remaining tasks without prematurely saving the run results.
