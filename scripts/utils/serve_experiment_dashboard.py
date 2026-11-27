# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
# ruff: noqa: E402

"""Serve the live baseline experiment dashboard on localhost.

Usage: .venv/bin/python scripts/utils/serve_experiment_dashboard.py --port 8765
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils import monitor_experiments as monitor
from scripts.utils.report_baseline_results import (
    DEFAULT_INITIAL_ROOT,
    DEFAULT_MONITOR_CONFIG,
    build_report,
)

HTML = Path(__file__).with_name("experiment_dashboard.html")
KST = timezone(timedelta(hours=9))
PROCESS_SCRIPTS = {
    "run_experiment.py",
    "evaluate_test_checkpoints.py",
    "evaluate_isolated.py",
    "retry_terminalbench2_test.py",
    "evaluate_bfcl_checkpoints.py",
    "run_isolated_baselines.py",
}


def running_processes() -> list[dict]:
    """Expose only experiment identifiers, never command arguments or environment variables."""
    rows = []
    for process in monitor.processes():
        if process["script"] not in PROCESS_SCRIPTS:
            continue
        flags = process["flags"]
        target = next(
            (flags[key] for key in ("--worker", "--worker-run", "--output-dir", "--output-root") if key in flags), ""
        )
        source = " ".join(str(value) for value in flags.values())
        method = (
            "ps_st_pareto"
            if "ps_st_pareto" in source
            else "premise"
            if "premise" in source
            else "ace"
            if "ace" in source
            else None
        )
        rows.append(
            {
                "pid": process["pid"],
                "script": process["script"],
                "method": method,
                "target": Path(target).name if target else "—",
            }
        )
    return sorted(rows, key=lambda row: (row["script"], row["target"], row["pid"]))


class ReportCache:
    def __init__(self, monitor_config: Path, initial_root: Path, ttl: float = 20.0):
        self.monitor_config = monitor_config
        self.initial_root = initial_root
        self.ttl = ttl
        self.lock = threading.Lock()
        self.entries: dict[str, tuple[float, dict]] = {}

    def get(self, metric: str) -> dict:
        with self.lock:
            cached = self.entries.get(metric)
            if cached and time.monotonic() - cached[0] < self.ttl:
                return cached[1]
            report = build_report(self.monitor_config, self.initial_root, metric)
            self.entries[metric] = (time.monotonic(), report)
            return report


def make_handler(cache: ReportCache) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "ExgenticDashboard/1.0"

        def send_data(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:",
            )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path in ("/", "/index.html"):
                self.send_data(200, "text/html; charset=utf-8", HTML.read_bytes())
                return
            if parsed.path == "/healthz":
                self.send_data(200, "application/json; charset=utf-8", b'{"status":"ok"}')
                return
            if parsed.path != "/api/status":
                self.send_data(404, "application/json; charset=utf-8", b'{"error":"not found"}')
                return
            metric = parse_qs(parsed.query).get("metric", ["official"])[0]
            if metric not in ("official", "task-mean"):
                self.send_data(400, "application/json; charset=utf-8", b'{"error":"invalid metric"}')
                return
            try:
                payload = {
                    "report": cache.get(metric),
                    "processes": running_processes(),
                    "served_at": datetime.now(KST).isoformat(timespec="seconds"),
                }
                body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
                self.send_data(200, "application/json; charset=utf-8", body)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logging.exception("Unable to collect experiment status")
                body = json.dumps({"error": f"Status collection failed: {type(exc).__name__}"}).encode("utf-8")
                self.send_data(500, "application/json; charset=utf-8", body)

        def log_message(self, format: str, *args: object) -> None:
            logging.info("%s %s", self.address_string(), format % args)

    return DashboardHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--monitor-config", type=Path, default=DEFAULT_MONITOR_CONFIG)
    parser.add_argument("--initial-root", type=Path, default=DEFAULT_INITIAL_ROOT)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cache = ReportCache(args.monitor_config.resolve(), args.initial_root.resolve())
    with ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(cache)) as server:
        server.daemon_threads = True
        print(f"Experiment dashboard: http://127.0.0.1:{args.port}/", flush=True)
        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
