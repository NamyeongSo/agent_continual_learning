#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

set -euo pipefail

# Official benchmark grading is supplied only after each task has ended,
# for ACE's post-session reflection and playbook update.
export ACE_TRAINING_TIME=1
export NUM_TASKS="${NUM_TASKS:-5}"
export MAX_TOKENS="${MAX_TOKENS:-4096}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_experiment_qwen.sh" "$@"
