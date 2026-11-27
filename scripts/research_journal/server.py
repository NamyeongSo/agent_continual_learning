# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""A local research notebook with optional external viewing, using the standard library."""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import logging
import os
import re
import secrets
import shutil
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer
from typing import BinaryIO
from urllib.parse import parse_qs, quote, urlsplit

from .live_progress import LiveProgress
from .reports import backup_archive, markdown_archive, report_csv, report_html, report_latex, table_preview
from .result_tables import latex as results_latex
from .store import IMAGE_TYPES, MAX_FILE_BYTES, MAX_PREVIEW_BYTES, ROOT, JournalError, JournalStore

STATIC = Path(__file__).with_name("static")


class JournalHTTPServer(ThreadingHTTPServer):
    """Bind without reverse DNS, which can stall hosts with many container interfaces."""

    def server_bind(self) -> None:
        TCPServer.server_bind(self)
        self.server_name = self.server_address[0]
        self.server_port = self.server_address[1]


def allowed_hostname(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?", value):
        raise argparse.ArgumentTypeError("Provide a hostname or IPv4 address without a scheme, port or wildcard")
    return value.lower()


def make_handler(
    store: JournalStore, token: str | None = None, *, allowed_hosts: tuple[str, ...] = ()
) -> type[BaseHTTPRequestHandler]:
    csrf_token = token or secrets.token_urlsafe(32)
    live = LiveProgress(store)

    class JournalHandler(BaseHTTPRequestHandler):
        server_version = "ResearchJournal/1.0"

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(30)

        def headers_for(
            self, status: int, content_type: str, length: int, filename: str | None = None, artifact: bool = False
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "SAMEORIGIN")
            policy = (
                "sandbox; default-src 'none'; style-src 'unsafe-inline'"
                if artifact
                else "default-src 'self'; img-src 'self' data:; frame-src 'self'; object-src 'self'; "
                "base-uri 'none'; form-action 'self'; frame-ancestors 'self'"
            )
            self.send_header("Content-Security-Policy", policy)
            if filename:
                self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(filename, safe='')}")
            self.end_headers()

        def send_bytes(
            self,
            body: bytes,
            content_type: str = "application/json; charset=utf-8",
            status: int = 200,
            filename: str | None = None,
        ) -> None:
            self.headers_for(status, content_type, len(body), filename)
            self.wfile.write(body)

        def send_json(self, payload: object, status: int = 200) -> None:
            self.send_bytes(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"), status=status)

        def send_stream(
            self, stream: BinaryIO, content_type: str, filename: str | None = None, artifact: bool = False
        ) -> None:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(0)
            self.headers_for(200, content_type, size, filename, artifact)
            shutil.copyfileobj(stream, self.wfile)

        def check_request(self, write: bool) -> None:
            port = self.server.server_address[1]
            allowed = {f"{host}:{port}" for host in ("127.0.0.1", "localhost", *allowed_hosts)}
            if self.headers.get("Host", "").lower() not in allowed:
                raise JournalError("허용된 기록장 주소로 접속해 주세요.", 403)
            if write:
                if self.read_only():
                    raise JournalError("외부 접속은 조회 전용입니다. 기록 수정은 로컬 접속에서 가능합니다.", 403)
                origin = self.headers.get("Origin")
                if origin and origin not in {f"http://{host}" for host in allowed}:
                    raise JournalError("다른 사이트에서는 기록을 수정할 수 없습니다.", 403)
                if not hmac.compare_digest(self.headers.get("X-Journal-Token", ""), csrf_token):
                    raise JournalError("페이지를 새로고침한 뒤 다시 시도해 주세요.", 403)

        def read_only(self) -> bool:
            return not ipaddress.ip_address(self.client_address[0]).is_loopback

        def read_body(self, limit: int = 1024 * 1024) -> bytes:
            if self.headers.get("Transfer-Encoding"):
                raise JournalError("Content-Length가 있는 요청이 필요합니다.")
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError as exc:
                raise JournalError("요청 크기가 올바르지 않습니다.") from exc
            if length < 0:
                raise JournalError("Content-Length가 필요합니다.")
            if length > limit:
                self.close_connection = True
                raise JournalError(f"요청 크기는 최대 {limit // (1024 * 1024)} MiB입니다.", 413)
            body = self.rfile.read(length)
            if len(body) != length:
                raise JournalError("파일 전송이 완료되지 않았습니다.")
            return body

        def read_json(self) -> dict:
            try:
                payload = json.loads(self.read_body())
            except (ValueError, UnicodeError) as exc:
                raise JournalError("올바른 JSON이 필요합니다.") from exc
            if not isinstance(payload, dict):
                raise JournalError("JSON 객체가 필요합니다.")
            return payload

        def handle_request(self) -> None:
            try:
                self.check_request(self.command != "GET")
                parsed = urlsplit(self.path)
                parts = [part for part in parsed.path.split("/") if part]
                query = parse_qs(parsed.query, keep_blank_values=True)
                if self.command == "GET":
                    self.get_route(parts, query)
                else:
                    self.write_route(parts, query)
            except JournalError as exc:
                self.send_json({"error": str(exc)}, exc.status)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                logging.info("Client disconnected")
            except Exception:
                logging.exception("Research journal request failed")
                self.send_json({"error": "처리 중 오류가 발생했습니다. 서버 로그와 데이터 폴더를 확인해 주세요."}, 500)

        def get_route(self, parts: list[str], query: dict) -> None:
            static_files = {
                (): ("index.html", "text/html; charset=utf-8"),
                ("static", "app.js"): ("app.js", "text/javascript; charset=utf-8"),
                ("static", "style.css"): ("style.css", "text/css; charset=utf-8"),
                ("static", "favicon.svg"): ("favicon.svg", "image/svg+xml"),
                ("static", "live.js"): ("live.js", "text/javascript; charset=utf-8"),
                ("static", "checkpoint_charts.js"): ("checkpoint_charts.js", "text/javascript; charset=utf-8"),
                ("static", "live.css"): ("live.css", "text/css; charset=utf-8"),
            }
            if tuple(parts) in static_files:
                name, mime = static_files[tuple(parts)]
                self.send_bytes((STATIC / name).read_bytes(), mime)
            elif parts == ["healthz"]:
                self.send_json({"status": "ok"})
            elif parts == ["api", "info"]:
                remote = self.read_only()
                self.send_json(
                    {
                        "csrf_token": None if remote else csrf_token,
                        "read_only": remote,
                        "data_dir": "" if remote else str(store.data_dir),
                        "allowed_roots": [] if remote else [str(path) for path in store.allowed_roots],
                        "base_dir": "" if remote else str(ROOT),
                        "max_file_bytes": MAX_FILE_BYTES,
                    }
                )
            elif parts == ["api", "experiments"]:
                self.send_json({"experiments": store.list()})
            elif parts == ["api", "live-progress"]:
                self.send_json(live.snapshot())
            elif parts == ["api", "proposal-retrieval"]:
                self.send_json(live.proposal_details(
                    query.get("batch", [None])[0], query.get("run", [None])[0],
                    query.get("iteration", [None])[0]))
            elif parts == ["api", "live-results.tex"]:
                tables = live.snapshot()["result_tables"]
                table = next((table for table in tables if table["id"] == query.get("id", [None])[0]), None)
                if table is None:
                    raise JournalError("결과표를 찾을 수 없습니다.", 404)
                self.send_bytes(results_latex(table).encode("utf-8"), "application/x-tex; charset=utf-8",
                                filename="live-main-results.tex")
            elif parts == ["api", "backup"]:
                if self.read_only():
                    raise JournalError("전체 백업은 로컬 접속에서 가능합니다.", 403)
                with closing(backup_archive(store)) as stream:
                    self.send_stream(stream, "application/zip", "research-journal-backup.zip")
            elif parts == ["api", "export"]:
                self.export(query)
            elif len(parts) == 3 and parts[:2] == ["api", "experiments"]:
                self.send_json(store.get(parts[2]))
            elif len(parts) == 4 and parts[:2] == ["api", "experiments"] and parts[3] == "files":
                if self.read_only():
                    raise JournalError("서버 파일 탐색은 로컬 접속에서 가능합니다.", 403)
                self.send_json(store.scan_files(parts[2]))
            elif len(parts) == 6 and parts[:2] == ["api", "experiments"] and parts[3] == "artifacts":
                self.serve_artifact(parts, query)
            else:
                raise JournalError("페이지를 찾을 수 없습니다.", 404)

        def serve_artifact(self, parts: list[str], query: dict) -> None:
            artifact, path = store.artifact(parts[2], parts[4])
            if parts[5] == "preview":
                if path.suffix.lower() in {".csv", ".tsv"}:
                    self.send_json({"kind": "table", **table_preview(path)})
                    return
                with path.open("rb") as stream:
                    content = stream.read(MAX_PREVIEW_BYTES + 1)
                self.send_json(
                    {
                        "kind": "text",
                        "text": content[:MAX_PREVIEW_BYTES].decode("utf-8", errors="replace"),
                        "truncated": len(content) > MAX_PREVIEW_BYTES,
                    }
                )
            elif parts[5] == "file":
                suffix = path.suffix.lower()
                mime = IMAGE_TYPES.get(suffix, "application/pdf" if suffix == ".pdf" else "application/octet-stream")
                download = "download" in query or suffix not in {*IMAGE_TYPES, ".pdf"}
                with path.open("rb") as stream:
                    self.send_stream(stream, mime, artifact["name"] if download else None, artifact=True)
            else:
                raise JournalError("첨부 파일 경로를 찾을 수 없습니다.", 404)

        def export(self, query: dict) -> None:
            ids = list(dict.fromkeys(query.get("ids", [""])[0].split(",")))
            documents = (
                [store.get(experiment_id) for experiment_id in ids]
                if ids != [""]
                else [doc for doc in store.list() if doc["status"] != "archived"]
            )
            if not documents:
                raise JournalError("내보낼 실험을 선택해 주세요.")
            format_name = query.get("format", ["html"])[0]
            exporters = {
                "html": (report_html, "text/html; charset=utf-8", "html"),
                "csv": (report_csv, "text/csv; charset=utf-8", "csv"),
                "latex": (report_latex, "application/x-tex; charset=utf-8", "tex"),
            }
            if format_name == "markdown":
                with closing(markdown_archive(store, documents)) as stream:
                    self.send_stream(stream, "application/zip", "research-report-markdown.zip")
            elif format_name in exporters:
                exporter, mime, suffix = exporters[format_name]
                body = exporter(store, documents) if format_name == "html" else exporter(documents)
                self.send_bytes(body, mime, filename=f"research-report.{suffix}")
            else:
                raise JournalError("내보내기 형식이 올바르지 않습니다.")

        def write_route(self, parts: list[str], query: dict) -> None:
            if parts == ["api", "experiments"] and self.command == "POST":
                self.send_json(store.create(self.read_json()), 201)
            elif len(parts) == 3 and parts[:2] == ["api", "experiments"] and self.command == "PUT":
                self.send_json(store.update(parts[2], self.read_json()))
            elif len(parts) == 4 and parts[:2] == ["api", "experiments"] and self.command == "POST":
                experiment_id, action = parts[2:]
                if action == "notes":
                    self.send_json(store.add_note(experiment_id, self.read_json()), 201)
                elif action == "import":
                    self.send_json(store.import_artifact(experiment_id, self.read_json()), 201)
                elif action == "artifacts":
                    body = self.read_body(MAX_FILE_BYTES)
                    self.send_json(
                        store.add_artifact(
                            experiment_id, query.get("name", [""])[0], body, query.get("caption", [""])[0]
                        ),
                        201,
                    )
                else:
                    raise JournalError("작업을 찾을 수 없습니다.", 404)
            elif len(parts) == 5 and parts[:2] == ["api", "experiments"] and parts[3] == "artifacts":
                if self.command == "PUT":
                    payload = self.read_json()
                    if not isinstance(payload.get("caption"), str):
                        raise JournalError("캡션 문자열이 필요합니다.")
                    self.send_json(store.change_artifact(parts[2], parts[4], payload["caption"]))
                elif self.command == "DELETE":
                    self.send_json(store.change_artifact(parts[2], parts[4]))
                else:
                    raise JournalError("지원하지 않는 작업입니다.", 405)
            else:
                raise JournalError("작업을 찾을 수 없습니다.", 404)

        do_GET = handle_request  # noqa: N815 - stdlib HTTP handler protocol
        do_POST = handle_request  # noqa: N815
        do_PUT = handle_request  # noqa: N815
        do_DELETE = handle_request  # noqa: N815

        def log_message(self, format: str, *args: object) -> None:
            logging.info("%s %s", self.address_string(), format % args)

    return JournalHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8878)
    parser.add_argument("--host", default="127.0.0.1", help="Listen address; default: loopback only.")
    parser.add_argument("--allow-host", type=allowed_hostname, action="append", default=[],
                        help="Explicit external viewing hostname or IPv4 address (without port; repeatable).")
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".experiment-journal")
    parser.add_argument(
        "--allow-root",
        type=Path,
        action="append",
        default=[],
        help="Additional local folders allowed for artifact import (repeatable).",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.host not in {"127.0.0.1", "localhost"} and not args.allow_host:
        parser.error("External listening requires an explicit --allow-host")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store = JournalStore(args.data_dir, [ROOT, *args.allow_root])
    # Keep an existing browser's write session valid during a managed restart.
    token = os.environ.pop("EXGENTIC_JOURNAL_SESSION_TOKEN", None)
    handler = make_handler(store, token=token, allowed_hosts=tuple(args.allow_host))
    with JournalHTTPServer((args.host, args.port), handler) as server:
        server.daemon_threads = True
        print(f"Research journal: http://127.0.0.1:{args.port}/", flush=True)
        for host in args.allow_host:
            print(f"External viewing: http://{host}:{args.port}/#live", flush=True)
        print(f"Data: {store.data_dir}", flush=True)
        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
