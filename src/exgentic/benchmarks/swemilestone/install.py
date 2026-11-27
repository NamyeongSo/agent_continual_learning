# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Install pinned benchmark assets beside a local or SSH Docker engine."""

from __future__ import annotations

import argparse
import json

from .transport import MilestoneBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-host", help="SSH Docker endpoint; defaults to DOCKER_HOST")
    parser.add_argument("--backend-root", help="Installation directory on the Docker host")
    parser.add_argument("--repo", default="all")
    parser.add_argument("--pull-images", action="store_true")
    args = parser.parse_args()
    backend = MilestoneBackend(args.docker_host, args.backend_root)
    print(json.dumps(backend.call("install", timeout=1800), indent=2))
    if args.pull_images:
        print(json.dumps(backend.call("images", subset=args.repo, timeout=14400), indent=2))


if __name__ == "__main__":
    main()
