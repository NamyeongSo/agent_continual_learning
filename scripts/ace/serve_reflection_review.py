# SPDX-License-Identifier: Apache-2.0
"""Launch the saved ACE Reflection comparison UI without inference or experiment imports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ace.reflection_review import main

if __name__ == "__main__":
    main()
