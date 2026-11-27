#!/usr/bin/env bash
set -euo pipefail
out=outputs/ace_appworld_advantage_add_20261118
aud=outputs/ace_appworld_aud_accumulation_qwen38_20261117
test -f "$aud/STOP"
test -f "$aud/thinking/STOP"
while ps -p 79769,205687 -o stat= | rg -q '^[^Z]'; do
    sleep 5
done
exec .venv/bin/python -m scripts.ace.run_advantage_accumulation --output "$out"
