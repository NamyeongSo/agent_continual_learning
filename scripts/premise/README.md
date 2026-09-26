# PREMiSE benchmark training

Use the shared frozen task manifest to train PREMiSE on BFCL, AppWorld, and
Terminal-Bench 2.0. The pipeline compares two parent playbooks, reflects on
trajectories, mutates candidates, and replays action prefixes in fresh sessions.

```bash
export OPENAI_API_BASE=http://127.0.0.1:8006/v1
export OPENAI_API_KEY=EMPTY
export DOCKER_HOST=ssh://your-docker-host
bash scripts/premise/run_terminalbench2.sh
```

For a configuration check, add `--num-tasks 1 --dry-run`. Set
`PREMISE_TASK_SPLIT_MANIFEST` and `PREMISE_OUTPUT_DIR` to override the frozen
50/39 task split and output path. Validation uses frozen playbooks and does
not update them. See [Terminal-Bench setup](../../docs/terminalbench2.md).
