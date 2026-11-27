# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""SQLite records with immutable copies of user-selected artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import os
import re
import sqlite3
import uuid
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STATUSES = {"planned", "running", "completed", "blocked", "archived"}
TEXT_FIELDS = (
    "title",
    "purpose",
    "hypothesis",
    "method",
    "benchmark",
    "model",
    "seeds",
    "split",
    "budget",
    "git_commit",
    "command",
    "run_path",
    "summary",
    "limitations",
    "next_steps",
    "notes",
    "started_on",
)
ARTIFACT_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".pdf",
    ".csv",
    ".tsv",
    ".json",
    ".txt",
    ".md",
    ".tex",
    ".html",
    ".ipynb",
    ".yaml",
    ".yml",
    ".log",
    ".zip",
}
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_PREVIEW_BYTES = 128 * 1024


class JournalError(Exception):
    """An expected error safe to show in the UI."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def clean_text(value: Any, label: str, limit: int = 50000) -> str:
    if not isinstance(value, str) or len(value) > limit or "\x00" in value:
        raise JournalError(f"{label}: 문자열을 입력해 주세요. 최대 {limit:,}자입니다.")
    return value.strip()


def validate_fields(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise JournalError("JSON 객체가 필요합니다.")
    result = {}
    for field in TEXT_FIELDS:
        if field in payload:
            result[field] = clean_text(payload[field], field, 200 if field == "title" else 50000)
            if field in {"notes", "command"}:
                result[field] = payload[field]
    for field in ("title", "purpose"):
        if field in result and not result[field]:
            raise JournalError("실험 제목과 목적은 필수입니다.")
    if "status" in payload:
        if not isinstance(payload["status"], str) or payload["status"] not in STATUSES:
            raise JournalError("올바른 실험 상태를 선택해 주세요.")
        result["status"] = payload["status"]
    if "tags" in payload:
        tags = payload["tags"]
        if not isinstance(tags, list) or len(tags) > 30:
            raise JournalError("태그는 최대 30개까지 등록할 수 있습니다.")
        result["tags"] = list(dict.fromkeys(clean_text(tag, "태그", 100) for tag in tags if tag))
    if "metrics" in payload:
        metrics = payload["metrics"]
        if not isinstance(metrics, list) or len(metrics) > 100:
            raise JournalError("지표는 최대 100개까지 등록할 수 있습니다.")
        result["metrics"] = []
        keys = set()
        for item in metrics:
            if not isinstance(item, dict):
                raise JournalError("지표 형식이 올바르지 않습니다.")
            metric = {key: clean_text(item.get(key, ""), key, 1000) for key in ("name", "unit", "split", "source")}
            value = item.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise JournalError("지표 값에는 숫자 또는 문자열을 입력해 주세요.")
            if isinstance(value, (int, float)) and (abs(value) > 1e100 or not math.isfinite(value)):
                raise JournalError("지표 값에는 유한한 숫자가 필요합니다.")
            metric["value"] = clean_text(value, "지표 값", 1000) if isinstance(value, str) else value
            if not metric["name"] or metric["value"] == "":
                raise JournalError("지표 이름과 값은 필수입니다. 0도 유효한 결과입니다.")
            key = (metric["name"], metric["split"], metric["unit"])
            if key in keys:
                raise JournalError("같은 이름·split·단위의 지표가 중복되었습니다.")
            keys.add(key)
            result["metrics"].append(metric)
    return result


class JournalStore:
    def __init__(self, data_dir: Path, allowed_roots: list[Path] | None = None):
        self.data_dir = data_dir.expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.blobs = self.data_dir / "artifacts"
        self.blobs.mkdir(exist_ok=True, mode=0o700)
        self.db_path = self.data_dir / "journal.sqlite3"
        self.allowed_roots = [path.expanduser().resolve() for path in (allowed_roots or [ROOT])]
        with closing(self.connect()) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, document TEXT NOT NULL)")

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=15)

    @staticmethod
    def _get(db: sqlite3.Connection, experiment_id: str) -> dict:
        row = db.execute("SELECT document FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        if not row:
            raise JournalError("실험을 찾을 수 없습니다.", 404)
        return json.loads(row[0])

    @staticmethod
    def _write(db: sqlite3.Connection, document: dict) -> dict:
        db.execute(
            "INSERT OR REPLACE INTO experiments VALUES (?, ?)",
            (document["id"], json.dumps(document, ensure_ascii=False, allow_nan=False)),
        )
        return document

    def get(self, experiment_id: str) -> dict:
        with closing(self.connect()) as db:
            return self._get(db, experiment_id)

    def list(self) -> list[dict]:
        with closing(self.connect()) as db:
            documents = [json.loads(row[0]) for row in db.execute("SELECT document FROM experiments")]
        return sorted(documents, key=lambda item: item["updated_at"], reverse=True)

    def create(self, payload: dict) -> dict:
        fields = validate_fields(payload)
        if not fields.get("title") or not fields.get("purpose"):
            raise JournalError("실험 제목과 목적은 필수입니다.")
        timestamp = now()
        document = {
            **dict.fromkeys(TEXT_FIELDS, ""),
            "status": "planned",
            "tags": [],
            "metrics": [],
            **fields,
            "id": f"exp-{datetime.now(UTC):%Y%m%d}-{uuid.uuid4().hex[:8]}",
            "created_at": timestamp,
            "updated_at": timestamp,
            "revision": 1,
            "artifacts": [],
            "log": [],
        }
        with closing(self.connect()) as db, db:
            return self._write(db, document)

    def update(self, experiment_id: str, payload: dict) -> dict:
        fields = validate_fields(payload)
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            document = self._get(db, experiment_id)
            revision = payload.get("revision")
            if type(revision) is not int or revision != document["revision"]:
                raise JournalError("다른 창에서 기록이 수정되었습니다. 입력 내용을 복사한 뒤 새로고침해 주세요.", 409)
            document.update(fields)
            document.update(updated_at=now(), revision=document["revision"] + 1)
            return self._write(db, document)

    def add_note(self, experiment_id: str, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise JournalError("JSON 객체가 필요합니다.")
        text = clean_text(payload.get("text", ""), "진행 기록")
        if not text:
            raise JournalError("진행 기록을 입력해 주세요.")
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            document = self._get(db, experiment_id)
            document["log"].append({"id": uuid.uuid4().hex, "time": now(), "text": text})
            document.update(updated_at=now(), revision=document["revision"] + 1)
            return self._write(db, document)

    def add_artifact(
        self, experiment_id: str, name: str, content: bytes, caption: str = "", source_path: str = ""
    ) -> dict:
        self.get(experiment_id)
        name = clean_text(name, "파일 이름", 250)
        if not name or Path(name).name != name or "\\" in name:
            raise JournalError("파일 이름이 올바르지 않습니다.")
        caption = clean_text(caption, "캡션", 10000)
        if len(content) > MAX_FILE_BYTES:
            raise JournalError("파일 하나당 최대 크기는 64 MiB입니다.", 413)
        digest = hashlib.sha256(content).hexdigest()
        suffix = Path(name).suffix.lower()
        suffix = suffix if re.fullmatch(r"\.[a-z0-9]{1,10}", suffix) else ".bin"
        filename = f"{uuid.uuid4().hex}{suffix}"
        path = self.blobs / filename
        artifact = {
            "id": uuid.uuid4().hex,
            "name": name,
            "filename": filename,
            "mime": IMAGE_TYPES.get(suffix, mimetypes.guess_type(name)[0] or "application/octet-stream"),
            "size": len(content),
            "sha256": digest,
            "caption": caption,
            "source_path": source_path,
            "created_at": now(),
        }
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            document = self._get(db, experiment_id)
            if any(item["sha256"] == digest and item["name"] == name for item in document["artifacts"]):
                return document
            try:
                path.write_bytes(content)
                document["artifacts"].append(artifact)
                document.update(updated_at=now(), revision=document["revision"] + 1)
                self._write(db, document)
            except Exception:
                path.unlink(missing_ok=True)
                raise
        return document

    def artifact(self, experiment_id: str, artifact_id: str) -> tuple[dict, Path]:
        document = self.get(experiment_id)
        artifact = next((item for item in document["artifacts"] if item["id"] == artifact_id), None)
        if artifact is None:
            raise JournalError("첨부 파일을 찾을 수 없습니다.", 404)
        path = (self.blobs / artifact["filename"]).resolve()
        if not path.is_relative_to(self.blobs) or not path.is_file():
            raise JournalError("첨부 파일 사본이 없습니다. 백업 상태를 확인해 주세요.", 404)
        return artifact, path

    def change_artifact(self, experiment_id: str, artifact_id: str, caption: str | None = None) -> dict:
        if caption is not None:
            caption = clean_text(caption, "캡션", 10000)
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            document = self._get(db, experiment_id)
            item = next((item for item in document["artifacts"] if item["id"] == artifact_id), None)
            if item is None:
                raise JournalError("첨부 파일을 찾을 수 없습니다.", 404)
            if caption is None:
                document["artifacts"].remove(item)
            else:
                item["caption"] = caption
            document.update(updated_at=now(), revision=document["revision"] + 1)
            return self._write(db, document)

    def resolve_source(self, source: str, *, directory: bool = False) -> Path:
        source = clean_text(source, "경로", 4096)
        if not source:
            raise JournalError("실험 폴더 경로를 먼저 입력해 주세요.")
        original = Path(source).expanduser()
        path = (original if original.is_absolute() else ROOT / original).resolve()
        root = next((root for root in self.allowed_roots if path.is_relative_to(root)), None)
        if root is None or path.is_relative_to(self.data_dir):
            raise JournalError("허용된 실험 폴더 밖의 경로입니다. 서버의 --allow-root 설정을 확인해 주세요.", 403)
        sensitive = {"node_modules", "venv", "env", "id_rsa", "id_ed25519", "credentials", "secrets"}
        if any(part.startswith(".") or part.lower() in sensitive for part in path.relative_to(root).parts):
            raise JournalError("숨김 파일과 비공개 설정 경로는 가져올 수 없습니다.", 403)
        if path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
            raise JournalError("키 파일은 가져올 수 없습니다.", 403)
        if not (path.is_dir() if directory else path.is_file()):
            raise JournalError("경로를 찾을 수 없습니다. 이 서버가 실행되는 컴퓨터의 경로를 입력해 주세요.", 404)
        return path

    def import_artifact(self, experiment_id: str, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise JournalError("JSON 객체가 필요합니다.")
        path = self.resolve_source(payload.get("path", ""))
        with path.open("rb") as stream:
            content = stream.read(MAX_FILE_BYTES + 1)
        return self.add_artifact(experiment_id, path.name, content, payload.get("caption", ""), str(path))

    def scan_files(self, experiment_id: str) -> dict:
        document = self.get(experiment_id)
        root = self.resolve_source(document["run_path"], directory=True)
        rows = []
        visited = 0
        truncated = False
        for folder, directories, files in os.walk(root, followlinks=False):
            visited += 1
            current = Path(folder)
            depth = len(current.relative_to(root).parts)
            directories[:] = sorted(
                name
                for name in directories
                if not name.startswith(".")
                and name
                not in {
                    "node_modules",
                    "venv",
                    "env",
                    "__pycache__",
                }
                and not (current / name).is_symlink()
            )
            if depth >= 3:
                truncated |= bool(directories)
                directories[:] = []
            for name in sorted(files):
                path = current / name
                if name.startswith(".") or path.suffix.lower() not in ARTIFACT_EXTENSIONS:
                    continue
                try:
                    path = self.resolve_source(str(path))
                    if not path.is_relative_to(root):
                        continue
                    info = path.stat()
                except (OSError, JournalError):
                    continue
                rows.append({"path": str(path), "relative": str(path.relative_to(root)), "size": info.st_size})
                if len(rows) >= 500:
                    truncated = True
                    break
            if len(rows) >= 500 or visited >= 1500:
                truncated = True
                break
        return {"root": str(root), "files": rows, "truncated": truncated}
