# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Run the local research journal: uv run python scripts/utils/serve_research_journal.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research_journal.server import main

if __name__ == "__main__":
    main()
