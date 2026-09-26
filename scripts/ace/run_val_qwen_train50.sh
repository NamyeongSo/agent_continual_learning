#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Resume the 54 frozen ACE validation evaluations (12 task sessions at a time).

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
cd "$repo_root"
source .venv/bin/activate

exec python -u scripts/ace/evaluate_val_checkpoints.py --max-workers 12 "$@"
