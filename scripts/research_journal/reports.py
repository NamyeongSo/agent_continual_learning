# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Offline reports and consistent, portable backups."""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from pathlib import Path
from typing import BinaryIO

from .store import IMAGE_TYPES, JournalError, JournalStore, now

STATUS_LABELS = {"planned": "계획", "running": "진행 중", "completed": "완료", "blocked": "보류", "archived": "보관"}
CONDITIONS = (
    ("method", "방법"),
    ("benchmark", "벤치마크"),
    ("model", "모델"),
    ("seeds", "Seed"),
    ("split", "데이터 / split"),
    ("budget", "예산 / 설정"),
    ("git_commit", "Git commit"),
    ("started_on", "시작일"),
    ("run_path", "실험 경로"),
    ("command", "실행 명령"),
)
SECTIONS = (
    ("purpose", "목적"),
    ("hypothesis", "가설 / 비교할 변경"),
    ("summary", "핵심 결론"),
    ("limitations", "한계 / 주의사항"),
    ("next_steps", "다음 할 일"),
    ("notes", "상세 메모"),
)
MAX_REPORT_BYTES = 128 * 1024 * 1024


def table_preview(path: Path, limit: int = 100) -> dict:
    with path.open("rb") as stream:
        content = stream.read(128 * 1024 + 1)
    text = content[: 128 * 1024].decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text), delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
    rows = []
    truncated_columns = False
    try:
        for row in reader:
            truncated_columns |= len(row) > 30 or any(len(cell) > 4000 for cell in row)
            rows.append([cell[:4000] for cell in row[:30]])
            if len(rows) > limit:
                break
    except csv.Error:
        pass
    return {"rows": rows[:limit], "truncated": truncated_columns or len(rows) > limit or len(content) > 128 * 1024}


def html_table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def metric_keys(documents: list[dict]) -> list[tuple[str, str, str]]:
    return list(
        dict.fromkeys(
            (metric["name"], metric["split"], metric["unit"]) for doc in documents for metric in doc["metrics"]
        )
    )


def metric_matrix(documents: list[dict]) -> tuple[list[str], list[list]]:
    headers = ["지표", "Split", "단위", *(doc["title"] for doc in documents)]
    rows = []
    for name, split, unit in metric_keys(documents):
        row = [name, split, unit]
        for doc in documents:
            metric = next(
                (item for item in doc["metrics"] if (item["name"], item["split"], item["unit"]) == (name, split, unit)),
                None,
            )
            row.append(metric["value"] if metric is not None else "—")
        rows.append(row)
    return headers, rows


def artifact_export_name(artifact: dict) -> str:
    return f"artifacts/{artifact['id']}-{artifact['name']}"


def report_html(store: JournalStore, documents: list[dict]) -> bytes:
    total_size = sum(artifact["size"] for doc in documents for artifact in doc["artifacts"])
    if total_size > MAX_REPORT_BYTES:
        raise JournalError(
            "선택한 첨부 파일이 128 MiB를 넘습니다. 실험을 나누어 내보내거나 전체 백업을 이용해 주세요.", 413
        )
    parts = [
        "<!doctype html><html lang='ko'><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<meta http-equiv='Content-Security-Policy' content=\"default-src 'none'; "
        "img-src data:; style-src 'unsafe-inline'; base-uri 'none'\">",
        "<title>실험 기록 보고서</title><style>",
        "body{font:15px/1.7 system-ui,sans-serif;color:#17232e;max-width:1080px;margin:40px auto;padding:0 24px}"
        "h1,h2,h3{line-height:1.35}h2{margin-top:32px}article{border-top:2px solid #bbc6ce;margin:40px 0}"
        ".text,pre{white-space:pre-wrap;overflow-wrap:anywhere}small,.muted{color:#566573}"
        "table{border-collapse:collapse;min-width:65%;font-size:13px}th,td{border:1px solid #ccd4d9;"
        "padding:8px 12px;text-align:left;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere}"
        "th{background:#eef3f5}.table-wrap{overflow:auto}img{max-width:100%;max-height:650px;object-fit:contain}"
        "figure{margin:24px 0;padding:16px;border:1px solid #ccd4d9}a{color:#086a60}"
        "@media print{body{margin:0;max-width:none;font-size:11px}article{break-before:page}"
        "figure,table{break-inside:avoid}a{color:inherit}.table-wrap{overflow:visible}}",
        f"</style><h1>실험 기록 보고서</h1><p class='muted'>생성: {html.escape(now())} · {len(documents)}개 실험</p>",
        "<p>이미지·표·첨부 파일을 포함하는 독립 HTML입니다. 첨부 파일은 각 다운로드 링크로 저장할 수 있습니다.</p>",
    ]
    if len(documents) > 1:
        parts.append("<h2>지표 비교</h2>" + html_table(*metric_matrix(documents)))
        parts.append(
            "<h3>재현 조건 비교</h3>"
            + html_table(
                ["조건", *(doc["title"] for doc in documents)],
                [[label, *(doc[field] or "—" for doc in documents)] for field, label in CONDITIONS],
            )
        )
    for doc in documents:
        parts.append(
            f"<article><h2>{html.escape(doc['title'])}</h2><p class='muted'>"
            f"{html.escape(doc['id'])} · {STATUS_LABELS[doc['status']]} · "
            f"수정: {html.escape(doc['updated_at'])}</p>"
        )
        if doc["tags"]:
            parts.append(f"<p>태그: {html.escape(', '.join(doc['tags']))}</p>")
        for field, label in SECTIONS:
            if doc[field]:
                parts.append(f"<h3>{label}</h3><div class='text'>{html.escape(doc[field])}</div>")
        parts.append(
            "<h3>재현 정보</h3>"
            + html_table(["항목", "값"], [[label, doc[field]] for field, label in CONDITIONS if doc[field]])
        )
        if doc["metrics"]:
            parts.append(
                "<h3>결과 지표</h3>"
                + html_table(
                    ["지표", "값", "단위", "Split", "근거 / 출처"],
                    [[item[key] for key in ("name", "value", "unit", "split", "source")] for item in doc["metrics"]],
                )
            )
        for artifact in doc["artifacts"]:
            _, path = store.artifact(doc["id"], artifact["id"])
            content = base64.b64encode(path.read_bytes()).decode("ascii")
            mime = IMAGE_TYPES.get(path.suffix.lower(), "application/octet-stream")
            uri = f"data:{mime};base64,{content}"
            parts.append(f"<figure><h3>{html.escape(artifact['name'])}</h3>")
            if path.suffix.lower() in IMAGE_TYPES:
                parts.append(f'<img src="{uri}" alt="{html.escape(artifact["caption"], quote=True)}">')
            if path.suffix.lower() in {".csv", ".tsv"}:
                preview = table_preview(path)
                if preview["rows"]:
                    parts.append(html_table(preview["rows"][0], preview["rows"][1:]))
                if preview["truncated"]:
                    parts.append(
                        "<p class='muted'>미리보기 일부입니다. 전체 데이터는 첨부 파일을 내려받아 확인하세요.</p>"
                    )
            parts.append(
                f"<figcaption class='text'>{html.escape(artifact['caption'])}</figcaption>"
                f'<p><a href="{uri}" download="{html.escape(artifact["name"], quote=True)}">'
                f"첨부 파일 다운로드 · {artifact['size']:,} bytes</a></p>"
                f"<small class='text'>원본 경로: {html.escape(artifact['source_path'] or '브라우저 업로드')}"
                f"<br>SHA-256: {artifact['sha256']}</small></figure>"
            )
        if doc["log"]:
            parts.append("<h3>진행 기록</h3>")
            for entry in doc["log"]:
                parts.append(
                    f"<p><small>{html.escape(entry['time'])}</small></p>"
                    f"<div class='text'>{html.escape(entry['text'])}</div>"
                )
        parts.append("</article>")
    parts.append("</html>")
    return "".join(parts).encode("utf-8")


def md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", "")


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = [
        "| " + " | ".join(md_cell(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(md_cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def report_markdown(documents: list[dict]) -> str:
    parts = ["# 실험 기록 보고서", f"생성: {now()}"]
    if len(documents) > 1:
        parts.extend(["## 지표 비교", md_table(*metric_matrix(documents))])
    for doc in documents:
        parts.extend(
            [f"## {doc['title']}", f"{doc['id']} · {STATUS_LABELS[doc['status']]} · 수정: {doc['updated_at']}"]
        )
        if doc["tags"]:
            parts.append("태그: " + ", ".join(doc["tags"]))
        for field, label in SECTIONS:
            if doc[field]:
                parts.extend([f"### {label}", doc[field]])
        parts.extend(
            [
                "### 재현 정보",
                md_table(["항목", "값"], [[label, doc[field]] for field, label in CONDITIONS if doc[field]]),
            ]
        )
        if doc["metrics"]:
            parts.extend(
                [
                    "### 결과 지표",
                    md_table(
                        ["지표", "값", "단위", "Split", "근거 / 출처"],
                        [
                            [item[key] for key in ("name", "value", "unit", "split", "source")]
                            for item in doc["metrics"]
                        ],
                    ),
                ]
            )
        for item in doc["artifacts"]:
            # Angle brackets preserve spaces and Unicode in relative Markdown URLs.
            target = artifact_export_name(item).replace("<", "%3C").replace(">", "%3E")
            prefix = "!" if Path(item["name"]).suffix.lower() in IMAGE_TYPES else ""
            label = item["name"].replace("[", "\\[").replace("]", "\\]")
            parts.extend(
                [
                    f"{prefix}[{label}](<{target}>)",
                    item["caption"],
                    f"원본: {item['source_path'] or '브라우저 업로드'} · SHA-256: {item['sha256']}",
                ]
            )
        if doc["log"]:
            parts.append("### 진행 기록")
            for entry in doc["log"]:
                parts.extend([f"#### {entry['time']}", entry["text"]])
    return "\n\n".join(parts) + "\n"


def spreadsheet_cell(value: object) -> object:
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def report_csv(documents: list[dict]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    fields = [
        "id",
        "title",
        "status",
        "purpose",
        "summary",
        "method",
        "benchmark",
        "model",
        "seeds",
        "split",
        "budget",
        "git_commit",
        "run_path",
        "updated_at",
    ]
    metric_fields = ["name", "value", "unit", "split", "source"]
    writer.writerow([*fields, *(f"metric_{key}" for key in metric_fields)])
    for doc in documents:
        for metric in doc["metrics"] or [{}]:
            writer.writerow(
                [
                    spreadsheet_cell(value)
                    for value in [*(doc[field] for field in fields), *(metric.get(key, "") for key in metric_fields)]
                ]
            )
    return ("\ufeff" + output.getvalue()).encode("utf-8")


def latex_escape(value: object) -> str:
    mapping = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\n": " ",
    }
    return "".join(mapping.get(char, char) for char in str(value))


def report_latex(documents: list[dict]) -> bytes:
    headers, rows = metric_matrix(documents)
    lines = [
        "% UTF-8. For Korean text, use XeLaTeX/LuaLaTeX with an appropriate font/package.",
        "% Rows match metric name, split and unit. Missing measurements are shown as a dash.",
        r"\begin{tabular}{" + "l" * len(headers) + "}",
        r"\hline",
        " & ".join(latex_escape(cell) for cell in headers) + r" \\",
        r"\hline",
    ]
    lines.extend(" & ".join(latex_escape(cell) for cell in row) + r" \\" for row in rows)
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines).encode("utf-8")


def markdown_archive(store: JournalStore, documents: list[dict]) -> BinaryIO:
    stream = tempfile.TemporaryFile()
    try:
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("report.md", report_markdown(documents))
            for doc in documents:
                for artifact in doc["artifacts"]:
                    _, path = store.artifact(doc["id"], artifact["id"])
                    archive.write(path, artifact_export_name(artifact))
        stream.seek(0)
        return stream
    except Exception:
        stream.close()
        raise


def backup_archive(store: JournalStore) -> BinaryIO:
    stream = tempfile.TemporaryFile()
    try:
        with tempfile.TemporaryDirectory(prefix="research-journal-backup-") as folder:
            snapshot_path = Path(folder) / "journal.sqlite3"
            with closing(store.connect()) as source, closing(sqlite3.connect(snapshot_path)) as destination:
                source.backup(destination)
                documents = [json.loads(row[0]) for row in destination.execute("SELECT document FROM experiments")]
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.write(snapshot_path, "journal.sqlite3")
                archive.writestr("experiments.json", json.dumps(documents, ensure_ascii=False, indent=2))
                archive.writestr(
                    "README.txt",
                    "Research Journal backup\n"
                    "Stop the server. Extract into a NEW empty directory.\n"
                    "Run serve_research_journal.py --data-dir /path/to/extracted-directory\n"
                    "journal.sqlite3 and artifacts/ must stay together.\n",
                )
                filenames = {item["filename"] for doc in documents for item in doc["artifacts"]}
                for name in sorted(filenames):
                    path = (store.blobs / name).resolve()
                    if not path.is_relative_to(store.blobs) or not path.is_file():
                        raise JournalError("백업할 첨부 사본이 없습니다. 데이터 폴더를 확인해 주세요.", 500)
                    archive.write(path, f"artifacts/{name}")
        stream.seek(0)
        return stream
    except Exception:
        stream.close()
        raise
