# Experiment code snapshot · 2026-11-27

The `research/ace-appworld-code-preservation-20261127` branch preserves the local
Exgentic research code as a standalone checkout of
`NamyeongSo/agent_continual_learning`. The existing Sico working directory,
branch, remote and experiment output folders are not modified.

The snapshot includes ACE reflection/candidate evaluation, ADD-only and
advantage-filtered accumulation, model scheduling/serving helpers, actor budget
and AppWorld scoring support, offline audits, analysis/figure generators, and
their saved configuration and focused tests. Existing PREMiSE, benchmark and
research-journal work in the same workspace is preserved as source as well.

The preserved ad hoc scripts and file hashes are indexed in
`research_snapshot/20261127/`. See its README for data dependencies and
historical-source limitations.

No raw rollout logs, experiment result packages, learned playbooks, model
weights, environments, installations or private Docker handoff files are part
of this snapshot. Large local histories remain at their original locations.
Ignore rules and an added-file size check protect future commits from those
artifacts; model inference is not part of validating this preservation branch.
