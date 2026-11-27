# SPDX-License-Identifier: Apache-2.0
"""Serve the local, read-only experiment protocol and instruction catalog."""

from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .build import DEFAULT_OUTPUT, clean, read

STATIC = Path(__file__).with_name("static")


def runtime_snapshot(path):
    """A running experiment can briefly expose an incomplete JSON write."""
    try:
        return clean(read(path))
    except (OSError, json.JSONDecodeError) as error:
        return {"phase": "unavailable", "read_error": type(error).__name__}


class Atlas:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.settings = read(self.output / "settings.json")
        self.tasks = read(self.output / "tasks.json")
        self.task_map = {(t["benchmark"], t["id"]): t for t in self.tasks}
        self.sources = {s["id"]: s for s in self.settings["sources"]}
        self.searches = {
            key: "\n".join(
                [t["id"], t["instruction"], json.dumps(t["meta"], ensure_ascii=False), *(r["text"] for r in t["turns"])]
            ).casefold()
            for key, t in self.task_map.items()
        }
        self.live_cache = None
        self.live_at = 0
        self.live_lock = threading.Lock()
        from .llm_calls import CallBrowser

        self.calls = CallBrowser()

    def catalog(self, query):
        benchmark = query.get("benchmark", [""])[0]
        split = query.get("split", [""])[0]
        text = query.get("q", [""])[0].casefold().strip()
        try:
            page = max(1, min(1000, int(query.get("page", ["1"])[0])))
        except ValueError:
            page = 1
        selected = query.get("selected", ["all"])[0]
        rows = [
            t
            for t in self.tasks
            if (not benchmark or t["benchmark"] == benchmark)
            and (not split or t["split"] == split)
            and (selected != "selected" or t["split"] != "unused")
            and (not text or text in self.searches[(t["benchmark"], t["id"])])
        ]
        rows.sort(key=lambda r: (r["benchmark"], r["id"]))
        return {
            "total": len(rows),
            "page": page,
            "page_size": 40,
            "items": [
                {k: r[k] for k in ("benchmark", "id", "split", "meta")}
                | {"preview": r["instruction"][:210], "turn_count": len(r["turns"])}
                for r in rows[(page - 1) * 40 : page * 40]
            ],
        }

    def live(self):
        with self.live_lock:
            if self.live_cache is not None and time.monotonic() - self.live_at < 15:
                return self.live_cache
            cohort = Path(self.settings["cohort"])
            result = {"checked_at": datetime.now(UTC).isoformat(), "cohort": {}, "runs": {}, "services": {}}
            for name in ("status", "progress"):
                path = cohort / f"{name}.json"
                if path.exists():
                    result["cohort"][name] = runtime_snapshot(path)
            for run in self.settings["runs"]:
                path = Path(run["output"]) / "job_status.json"
                result["runs"][run["id"]] = runtime_snapshot(path) if path.exists() else {"phase": "pending"}
            for name, url in (
                ("vllm", "http://127.0.0.1:8006/health"),
                ("parser", "http://127.0.0.1:18006/healthz"),
                ("reasoning_bridge", "http://127.0.0.1:18007/healthz"),
            ):
                try:
                    with urllib.request.urlopen(url, timeout=1.5) as response:
                        result["services"][name] = {"online": response.status == 200}
                except (OSError, urllib.error.URLError):
                    result["services"][name] = {"online": False}
            self.live_cache, self.live_at = result, time.monotonic()
            return result

    def task_csv(self):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(["benchmark", "task_id", "split", "instruction", "user_turns", "metadata"])
        for task in self.tasks:
            writer.writerow(
                [
                    task["benchmark"],
                    task["id"],
                    task["split"],
                    "\n\n".join(t["text"] for t in task["turns"]) if task["turns"] else task["instruction"],
                    len(task["turns"]),
                    json.dumps(task["meta"], ensure_ascii=False),
                ]
            )
        return ("\ufeff" + stream.getvalue()).encode()


def make_handler(atlas):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            if args and str(args[1] if len(args) > 1 else "").startswith("5"):
                super().log_message(format, *args)

        def reply(self, content, content_type="application/json; charset=utf-8", status=200, filename=None):
            if not isinstance(content, bytes):
                content = json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            url = urllib.parse.urlsplit(self.path)
            path = urllib.parse.unquote(url.path)
            query = urllib.parse.parse_qs(url.query)
            if path in {"/", "/index.html", "/app.js", "/style.css", "/llm_calls.js"}:
                file = STATIC / ("index.html" if path in {"/", "/index.html"} else path[1:])
                self.reply(file.read_bytes(), (mimetypes.guess_type(file.name)[0] or "text/plain") + "; charset=utf-8")
            elif path == "/api/settings":
                self.reply(atlas.settings)
            elif path == "/api/health":
                self.reply(
                    {
                        "ok": True,
                        "tasks": len(atlas.tasks),
                        "runs": len(atlas.settings["runs"]),
                        "generated_at": atlas.settings["generated_at"],
                    }
                )
            elif path == "/api/live":
                self.reply(atlas.live())
            elif path == "/api/shared-reflection":
                from .shared_reflection import snapshot

                self.reply(snapshot())
            elif path == "/api/checkpoint-evaluation":
                from .checkpoint_progress import snapshot

                self.reply(snapshot())
            elif path.startswith("/api/llm/"):
                try:
                    identity = query.get("job", [""])[0]
                    source_id = query.get("source", [""])[0]
                    if path == "/api/llm/jobs":
                        value = atlas.calls.jobs()
                    elif path == "/api/llm/sources":
                        value = atlas.calls.catalog(query)
                    elif path == "/api/llm/records":
                        value = atlas.calls.records(identity, source_id)
                    elif path == "/api/llm/call":
                        value = atlas.calls.detail(identity, source_id, int(query.get("line", ["0"])[0]))
                    else:
                        raise KeyError("Unknown model log endpoint")
                    self.reply(value, filename="llm-call.json" if query.get("download") == ["1"] else None)
                except (KeyError, FileNotFoundError):
                    self.reply({"error": "Recorded call or experiment not found"}, status=404)
                except (ValueError, TypeError):
                    self.reply({"error": "Invalid model log query or malformed record"}, status=400)
                except OSError:
                    self.reply({"error": "Model log is temporarily unavailable"}, status=503)
            elif path == "/api/tasks":
                self.reply(atlas.catalog(query))
            elif path.startswith("/api/task/"):
                parts = path.split("/")
                task = atlas.task_map.get(tuple(parts[3:])) if len(parts) == 5 else None
                self.reply(task if task else {"error": "Task not found"}, status=200 if task else 404)
            elif path.startswith("/api/source/"):
                source = atlas.sources.get(path.rsplit("/", 1)[-1])
                self.reply(source if source else {"error": "Source not found"}, status=200 if source else 404)
            elif path == "/download/settings.json":
                self.reply(atlas.settings, filename="experiment-settings.json")
            elif path == "/download/tasks.csv":
                self.reply(atlas.task_csv(), "text/csv; charset=utf-8", filename="task-instructions.csv")
            elif path == "/favicon.ico":
                self.reply(b"", "image/x-icon", status=204)
            else:
                self.reply({"error": "Not found"}, status=404)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8880)
    args = parser.parse_args()
    atlas = Atlas(args.data_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(atlas))
    print(
        json.dumps(
            {"url": f"http://{args.host}:{args.port}", "data_dir": str(atlas.output), "tasks": len(atlas.tasks)},
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
