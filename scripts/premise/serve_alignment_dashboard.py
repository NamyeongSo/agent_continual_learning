# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Serve recorded PREMiSE alignment judgments and persist binary human reviews locally.

Usage: .venv/bin/python scripts/premise/serve_alignment_dashboard.py --port 8766
Only Python's standard library is required. No model or benchmark calls are made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).parent
DEFAULT_RUN = "premise_completion_contract_restart_20261003_0557/s42_interleaved_t3"
AXES = [
    {
        "id": "evidence_grounding",
        "name": "근거 충실성",
        "question": "판정의 근거가 실제 기록된 행동과 관찰에 있는가?",
        "pass": "실제 action·observation으로 설명을 확인할 수 있고, 누락된 사실을 지어내지 않는다.",
        "fail": "관찰에 없는 행동·결과를 주장하거나, 생략·실행 오류를 성공의 근거로 사용한다.",
    },
    {
        "id": "intent_interpretation",
        "name": "의도 해석",
        "question": "reflection의 expected_behavior를 정확히 해석했는가?",
        "pass": "기대 행동의 조건과 핵심 절차를 정확히 이해하고 replay와 비교한다.",
        "fail": "기대 행동을 잘못 이해하거나, 핵심 조건을 빼고 다른 행동을 평가한다.",
    },
    {
        "id": "behavioral_focus",
        "name": "행동 중심성",
        "question": "원본 span과 replay의 실제 행동을 비교해 판정했는가?",
        "pass": "복원 지점과 이후 행동을 구분하고, 성공 점수·playbook 문구에 기대지 않는다.",
        "fail": "최종 성공 점수나 playbook 문구만으로 판정하거나, prefix를 변경된 행동으로 오인한다.",
    },
    {
        "id": "score_calibration",
        "name": "점수 적정성",
        "question": "judge의 0 / 0.5 / 1 점수가 실제 일치 정도에 적절한가?",
        "pass": "불일치는 0, 부분 일치는 0.5, 일치는 1로 납득할 수 있게 구분했다.",
        "fail": "불일치를 일치로 보거나, 부분 일치를 완전 일치로 보는 등 점수 수준이 부적절하다.",
    },
    {
        "id": "explanation_consistency",
        "name": "설명 일관성",
        "question": "점수·label·rationale이 서로 일치하고 모순이 없는가?",
        "pass": "0=not_aligned, 0.5=partial, 1=aligned이며, 설명도 그 판정을 뒷받침한다.",
        "fail": "점수와 label이 다르거나, rationale 안의 주장 또는 결론이 서로 모순된다.",
    },
]
AXIS_IDS = {axis["id"] for axis in AXES}
PRUNED = {
    "sessions", "llm_calls", "checkpoints", "evaluation", "test_evaluation", "val_evaluation",
    "attempts", "logs", ".git", ".venv", "node_modules", "dataset", "discovery", "recovery",
}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def text_value(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


def has_judgments(artifact: dict) -> bool:
    return any(
        "alignment" in result
        for candidate in artifact.get("candidates", [])
        for result in [candidate.get("current_result", {}), *candidate.get("past_results", [])]
    )


def discover_runs(output_root: Path) -> list[dict]:
    """Discover actual alignment runs without traversing benchmark session trees."""
    runs = []
    for parent, dirs, _files in os.walk(output_root):
        dirs[:] = [name for name in dirs if name not in PRUNED]
        if Path(parent).name != "iterations":
            continue
        dirs[:] = []
        paths = sorted(Path(parent).glob("iter_*.json"))
        for sample in reversed(paths[-5:]):
            try:
                if not has_judgments(read_json(sample)):
                    continue
                run = Path(parent).parent.resolve()
                relative = run.relative_to(output_root).as_posix()
                runs.append({
                    "id": digest(relative)[:16], "name": relative, "path": run,
                    "iterations": len(paths), "modified": paths[-1].stat().st_mtime,
                })
                break
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return sorted(runs, key=lambda item: item["modified"], reverse=True)


def memory_key(value: dict) -> tuple[str, str]:
    return str(value.get("benchmark", "")), str(value.get("task_id", ""))


def reconstructed_judge_prompt(source: dict | None, reflection: dict | None, replay: dict) -> str | None:
    """Render the current judge template; historical request logs may use older templates."""
    if source is None or reflection is None:
        return None

    def actions_text(actions: list[dict]) -> str:
        return "\n\n".join(
            f"### action[{index}]\n{json.dumps(item['action'], ensure_ascii=False, default=str)}\n"
            f"Observation: {json.dumps(item.get('observation'), ensure_ascii=False, default=str)}"
            for index, item in enumerate(actions)
        ) or "(no actions)"

    span = reflection["span"]
    return (
        "Judge whether the candidate playbook changed the agent's actual replay behavior in the direction "
        "stated by the reflection. Judge behavior, not playbook wording or final score. Compare the original "
        "span with the continuation from the restored state.\n\n"
        f"Benchmark: {source['benchmark']}\nTask: {source['task']}\n"
        f"Reflection analysis: {reflection['analysis']}\nExpected behavior: {reflection['expected_behavior']}\n"
        f"Original span: {actions_text(source['actions'][span['start']:span['end']])}\n"
        f"Candidate continuation: {actions_text(replay['actions'])}\n\n"
        'Return JSON {"score":0|0.5|1,"label":"not_aligned|partial|aligned","rationale":"brief reason"}. '
        "1 means aligned, 0.5 partial, 0 not aligned.\n\n"
        "Output exactly one JSON object; no Markdown or commentary. Use the numeric score 0, 0.5, or 1 "
        "and the matching label not_aligned, partial, or aligned. Example: "
        '{"score":0.5,"label":"partial","rationale":"brief evidence-based reason"}. '
        "Do not use a literal pipe-separated list as a value."
    )


def source_reference(item: dict) -> dict | None:
    if isinstance(item.get("trace"), dict):
        return {"trace": item["trace"], "reflection": item.get("reflection", {})}
    if item.get("trace_artifact") is not None and item.get("parent_index") is not None:
        return {key: item.get(key) for key in ("trace_artifact", "parent_index", "reflection")}
    return None


class Dataset:
    """Keep lightweight row indices; load complete evidence only on selection."""

    def __init__(self, run: dict):
        self.run = run
        self.rows: list[dict] = []
        self.locations: dict[str, dict] = {}
        self.warnings: list[str] = []
        self.source_hashes: dict[Path, str] = {}
        self._build()

    def _build(self) -> None:
        root = self.run["path"]
        memory: dict[tuple[str, str], dict] = {}
        state_memory: dict[tuple[str, str], list[dict]] = {}
        try:
            state = read_json(root / "state.json")
            for group in state.get("states", {}).values():
                for item in group.get("memory", []):
                    state_memory.setdefault(memory_key(item), []).append(item)
        except (OSError, ValueError):
            pass
        for path in sorted((root / "iterations").glob("iter_*.json")):
            try:
                raw = path.read_bytes()
                artifact = json.loads(raw)
                fingerprint = digest(raw)
                iteration = int(artifact["iteration"])
                past = artifact.get("past_sample", [])
                for candidate in artifact.get("candidates", []):
                    current = candidate.get("current_result", {})
                    if "alignment" in current:
                        self._append(path, artifact, candidate, current, "current", None, None, fingerprint)
                    for index, result in enumerate(candidate.get("past_results", [])):
                        if "alignment" not in result:
                            continue
                        sample = past[index] if index < len(past) else result.get("replay", {})
                        reference = memory.get(memory_key(sample))
                        if reference is None:
                            # Restarts can refer to accepted memory outside the local iteration prefix.
                            entries = state_memory.get(memory_key(sample), [])
                            eligible = [entry for entry in entries if self._origin_iteration(entry) < iteration]
                            if eligible:
                                reference = source_reference(max(eligible, key=self._origin_iteration))
                        self._append(path, artifact, candidate, result, "past", index, reference, fingerprint)
                selected = artifact.get("selection", {}).get("selected", [])
                if selected:
                    primary = next((c for c in artifact.get("candidates", []) if c["id"] == selected[0]), None)
                    if primary and primary.get("applied"):
                        memory[memory_key(artifact)] = {
                            "trace_artifact": path.relative_to(root).as_posix(),
                            "parent_index": primary["parent_rank"] - 1,
                            "reflection": primary["reflection"],
                        }
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.warnings.append(f"{path.name}: {type(exc).__name__}; 읽을 수 있는 판정만 포함했습니다.")

    @staticmethod
    def _origin_iteration(item: dict) -> int:
        match = re.search(r"iter_(\d+)", str(item.get("trace_artifact", "")))
        if match:
            return int(match[1])
        try:
            return int(str(item.get("memory_id", "0")).split("|", 1)[0])
        except ValueError:
            return 0

    def _append(self, path, artifact, candidate, result, scope, index, reference, fingerprint) -> None:
        judge = result["alignment"]
        replay = result.get("replay", {})
        source_hash = ""
        if reference:
            source_hash = digest(json.dumps(reference, sort_keys=True, ensure_ascii=False))
            if reference.get("trace_artifact"):
                source_path = (self.run["path"] / reference["trace_artifact"]).resolve()
                if source_path.is_relative_to(self.run["path"]) and source_path.is_file():
                    if source_path not in self.source_hashes:
                        self.source_hashes[source_path] = digest(source_path.read_bytes())
                    source_hash += self.source_hashes[source_path]
        identity = f"{self.run['id']}|{path.name}|{candidate['id']}|{scope}|{index}|{fingerprint}|{source_hash}"
        case_id = digest(identity)
        row = {
            "id": case_id, "iteration": artifact["iteration"], "candidate": candidate["id"],
            "scope": scope, "past_index": index, "benchmark": replay.get("benchmark", artifact["benchmark"]),
            "task_id": str(replay.get("task_id", artifact["task_id"])),
            "judge_score": judge.get("score"), "judge_label": judge.get("label", ""),
            "rationale": judge.get("rationale", ""), "valid_candidate": bool(candidate.get("valid")),
            "selected": candidate["id"] in artifact.get("selection", {}).get("selected", []),
            "source_available": scope == "current" or reference is not None,
            "artifact": path.relative_to(self.run["path"]).as_posix(),
        }
        self.rows.append(row)
        self.locations[case_id] = {
            "row": row, "path": path, "reference": reference, "fingerprint": fingerprint,
        }

    def case(self, case_id: str) -> dict:
        location = self.locations[case_id]
        row = location["row"]
        raw = location["path"].read_bytes()
        if digest(raw) != location["fingerprint"]:
            raise ValueError("실험 기록이 변경되었습니다. 데이터 새로고침 후 다시 선택하세요.")
        artifact = json.loads(raw)
        candidate = next(item for item in artifact["candidates"] if item["id"] == row["candidate"])
        warnings = []
        if row["scope"] == "current":
            source = artifact["parents"][candidate["parent_rank"] - 1]
            reflection = candidate["reflection"]
            result = candidate["current_result"]
            source_artifact = row["artifact"]
        else:
            result = candidate["past_results"][row["past_index"]]
            reference = location["reference"]
            source, reflection, source_artifact = None, None, None
            if reference:
                reflection = reference.get("reflection")
                if "trace" in reference:
                    source = reference["trace"]
                    source_artifact = "state.json (embedded trace)"
                else:
                    source_path = (self.run["path"] / reference["trace_artifact"]).resolve()
                    if not source_path.is_relative_to(self.run["path"]):
                        raise ValueError("Memory artifact must stay inside its experiment run")
                    try:
                        source_raw = source_path.read_bytes()
                        if digest(source_raw) != self.source_hashes.get(source_path):
                            raise ValueError("과거 원본이 변경되었습니다. 데이터 새로고침이 필요합니다.")
                        original = json.loads(source_raw)
                        source = original["parents"][reference["parent_index"]]
                        source_artifact = reference["trace_artifact"]
                    except (OSError, KeyError, IndexError, TypeError):
                        warnings.append("과거 원본 artifact를 읽을 수 없습니다.")
            if source and memory_key(source) != memory_key(result.get("replay", {})):
                source, reflection = None, None
                warnings.append("과거 원본의 task 식별자가 replay와 다릅니다. 근거를 연결하지 않았습니다.")
            if not source or not reflection:
                warnings.append("당시 사용된 과거 원본·reflection을 복원하지 못했습니다. 증거 부족을 메모해 주세요.")
        span = (reflection or {}).get("span", {})
        return {
            **row, "judge": result["alignment"], "reflection": reflection, "source": source,
            "span": span, "replay": result.get("replay", {}), "source_artifact": source_artifact,
            "official_score": result.get("official_score"), "mutation_summary": candidate.get("mutation_summary", ""),
            "applied": candidate.get("applied", []), "warnings": warnings,
            "reconstructed_prompt": reconstructed_judge_prompt(source, reflection, result.get("replay", {})),
        }


class ReviewStore:
    """Atomically persist completed and partial annotations, including explicit zeroes."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data = {"schema_version": 1, "axes": AXES, "reviews": {}}
        if path.exists():
            self.data = read_json(path)  # Fail visibly instead of overwriting corrupt annotations.
            if self.data.get("schema_version") != 1 or not isinstance(self.data.get("reviews"), dict):
                raise ValueError("Unsupported review file; existing annotations were preserved")

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.data))

    def save(self, case_id: str, run: dict, row: dict, review: dict) -> dict:
        axes = review.get("axes")
        if not isinstance(axes, dict) or set(axes) != AXIS_IDS:
            raise ValueError("다섯 평가 축을 모두 포함해야 합니다.")
        if any(value is not None and (type(value) is not int or value not in (0, 1)) for value in axes.values()):
            raise ValueError("축 점수는 0, 1, 또는 미평가(null)만 허용됩니다.")
        checked, notes = review.get("checked"), review.get("notes", "")
        if type(checked) is not bool or not isinstance(notes, str) or len(notes) > 20000:
            raise ValueError("검증 체크 또는 메모 형식이 올바르지 않습니다.")
        if checked and any(value is None for value in axes.values()):
            raise ValueError("검증 완료 전에 다섯 축을 모두 평가해 주세요.")
        saved = {
            "axes": axes, "checked": checked, "notes": notes,
            "updated_at": datetime.now(UTC).isoformat(), "run_id": run["id"], "run": run["name"],
            "benchmark": row["benchmark"], "task_id": row["task_id"], "iteration": row["iteration"],
            "candidate": row["candidate"], "scope": row["scope"], "past_index": row["past_index"],
            "judge_score": row["judge_score"], "artifact": row["artifact"],
        }
        with self.lock:
            updated = {**self.data, "reviews": {**self.data["reviews"], case_id: saved}}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                                 prefix=".reviews-", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    json.dump(updated, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(self.path)
                self.data = updated
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return saved


class Application:
    def __init__(self, output_root: Path, review_file: Path):
        self.output_root = output_root
        self.store = ReviewStore(review_file)
        self.lock = threading.Lock()
        self.runs = {run["id"]: run for run in discover_runs(output_root)}
        self.datasets: dict[str, Dataset] = {}

    def catalog(self) -> dict:
        preferred = next((run["id"] for run in self.runs.values() if run["name"] == DEFAULT_RUN), None)
        return {
            "runs": [{key: value for key, value in run.items() if key != "path"} for run in self.runs.values()],
            "default_run": preferred or next(iter(self.runs), None), "axes": AXES,
            "review_file": str(self.store.path),
        }

    def dataset(self, run_id: str, refresh: bool = False) -> Dataset:
        with self.lock:
            if run_id not in self.runs:
                raise KeyError("Unknown experiment run")
            if refresh or run_id not in self.datasets:
                self.datasets[run_id] = Dataset(self.runs[run_id])
            return self.datasets[run_id]


def make_handler(app: Application) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "PremiseAlignmentAudit/1.0"

        def send(self, status: int, data: object, content_type: str = "application/json; charset=utf-8") -> None:
            body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                             "script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            try:
                assets = {
                    "/": ("alignment_dashboard.html", "text/html; charset=utf-8"),
                    "/alignment_dashboard.js": ("alignment_dashboard.js", "application/javascript; charset=utf-8"),
                    "/alignment_dashboard.css": ("alignment_dashboard.css", "text/css; charset=utf-8"),
                }
                if url.path in assets:
                    name, mime = assets[url.path]
                    self.send(200, (ASSETS / name).read_bytes(), mime)
                elif url.path == "/api/catalog":
                    self.send(200, app.catalog())
                elif url.path == "/api/run":
                    dataset = app.dataset(query["run"][0], query.get("refresh") == ["1"])
                    reviews = app.store.snapshot()["reviews"]
                    self.send(200, {
                        "run": dataset.run["name"], "records": dataset.rows,
                        "reviews": {key: reviews[key] for key in dataset.locations if key in reviews},
                        "warnings": dataset.warnings,
                    })
                elif url.path == "/api/case":
                    self.send(200, app.dataset(query["run"][0]).case(query["id"][0]))
                elif url.path == "/api/export":
                    self.send(200, app.store.snapshot())
                elif url.path == "/api/health":
                    self.send(200, {"status": "ok", "runs": len(app.runs)})
                else:
                    self.send(404, {"error": "Not found"})
            except (KeyError, IndexError):
                self.send(404, {"error": "해당 실험 또는 판정을 찾을 수 없습니다."})
            except ValueError as exc:
                self.send(409, {"error": str(exc)})
            except OSError:
                logging.exception("Could not read dashboard data")
                self.send(500, {"error": "로컬 데이터 파일을 읽을 수 없습니다."})

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/review":
                self.send(404, {"error": "Not found"})
                return
            origin = self.headers.get("Origin")
            if ((origin and urlsplit(origin).netloc != self.headers.get("Host"))
                    or self.headers.get("Sec-Fetch-Site") == "cross-site"):
                self.send(403, {"error": "Cross-origin writes are not allowed"})
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.send(415, {"error": "JSON body required"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 100000:
                    raise ValueError("Review body is empty or too large")
                body = json.loads(self.rfile.read(length))
                dataset = app.dataset(body["run"])
                row = dataset.locations[body["id"]]["row"]
                dataset.case(body["id"])  # Do not attach a review to evidence modified since loading.
                saved = app.store.save(body["id"], dataset.run, row, body["review"])
                self.send(200, {"review": saved})
            except (KeyError, TypeError, AttributeError):
                self.send(400, {"error": "판정 ID 또는 평가 형식이 올바르지 않습니다."})
            except (ValueError, UnicodeError) as exc:
                self.send(400, {"error": str(exc)})
            except OSError:
                logging.exception("Could not save human review")
                self.send(500, {"error": "저장에 실패했습니다. 변경 사항을 유지하고 다시 시도하세요."})

        def log_message(self, fmt: str, *args: object) -> None:
            logging.info("%s %s", self.address_string(), fmt % args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--review-file", type=Path, default=ROOT / "outputs/alignment_judge_audit/reviews.json")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = Application(args.output_root.resolve(), args.review_file.resolve())
    with ThreadingHTTPServer((args.host, args.port), make_handler(app)) as server:
        server.daemon_threads = True
        print(f"PREMiSE alignment audit: http://{args.host}:{args.port}/", flush=True)
        print(f"Experiments: {len(app.runs)} | Reviews: {app.store.path}", flush=True)
        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
