# SPDX-License-Identifier: Apache-2.0
"""Serve blind comparisons of selected ACE Reflections or their independent playbook additions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import json
import re
import secrets
import sqlite3
from collections import defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).with_name("reflection_review_static")
MODELS = (
    ("qwen38", "Qwen3.8 · 27B", "ace_appworld_advantage_add_20261118/plan.json"),
    ("qwen36", "Qwen3.6 · 35B-A3B", "ace_appworld_qwen36_add_only_20261118/advantage/plan.json"),
    ("qwen35", "Qwen3.5 · 9B", "ace_appworld_qwen35_9b_add_only_20261119/advantage/plan.json"),
    ("oss120", "GPT-OSS · 120B", "ace_appworld_gpt_oss120b_add_only_20261121/advantage/plan.json"),
    ("oss20", "GPT-OSS · 20B", "ace_appworld_gpt_oss20b_add_only_20261120/advantage/plan.json"),
    ("gemma26", "Gemma4 · 26B-A4B", "ace_appworld_gemma4_26b_a4b_it_wo_gt_train_only_20261124/advantage/plan.json"),
    ("gemma31", "Gemma4 · 31B", "ace_appworld_gemma4_31b_it_wo_gt_train_only_20261124/advantage/plan.json"),
)
STRATEGIES = ("best_advantage", "worst_advantage")
CONTENT_KINDS = ("reflection", "independent_rules")
RULE_PREFIX = re.compile(r"(?m)^\[([^\]\r\n]+)\][ \t]+helpful=\d+[ \t]+harmful=\d+[ \t]+::[ \t]?")


def now():
    return datetime.now(UTC).isoformat()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def added_rules(initial, updated):
    """Extract actual multiline bullet bodies from a saved update of the empty sectioned playbook."""
    if RULE_PREFIX.search(initial):
        raise ValueError("Independent rule comparison requires an empty learned playbook")
    headers = {line for line in initial.split("\n") if line.strip()}
    if any(not line.startswith("## ") for line in headers):
        raise ValueError("Unexpected initial playbook content")
    matches = list(RULE_PREFIX.finditer(updated))
    prefix = updated[:matches[0].start()] if matches else updated
    if any(line.strip() and line not in headers for line in prefix.split("\n")):
        raise ValueError("Unrecognized text outside actual playbook rules")
    result = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(updated)
        lines = updated[match.end():end].split("\n")
        while lines and (not lines[-1].strip() or lines[-1] in headers):
            lines.pop()
        result.append({"id": match[1], "content": "\n".join(lines)})
    if len({rule["id"] for rule in result}) != len(result):
        raise ValueError("Duplicate playbook rule IDs")
    return result


def independent_rules(original, candidate, arm):
    """Verify an isolated stage2 update of this exact task/Reflection and retain only applied rules."""
    paths = {}

    def load(path):
        raw = path.read_bytes()
        paths[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    reflection = load(original)
    result_path = original.parent / "result.json"
    result = load(result_path)
    if (result["stage"], result["task_id"], result["source_repeat"], result["arm"]) != (
            2, candidate["task_id"], candidate["source_repeat"], arm):
        raise ValueError("Independent update does not match the selected task, candidate, and GT setting")
    if Path(result["reflection_path"]).resolve() != original.resolve():
        raise ValueError("Independent update points to a different Reflection")
    protocol = load(original.parents[2] / "protocol.json")
    initial = protocol["initial_playbook"]
    if protocol["playbook_edit_mode"] != "add_only":
        raise ValueError("Independent update is not ADD-only")
    capture_path = Path(result["source_capture_path"])
    capture = load(capture_path)
    if paths[str(capture_path)] != reflection["source_capture_sha256"] or capture["playbook"] != initial:
        raise ValueError("Independent update did not start from the saved initial playbook")
    if reflection["reflection_for_curator"] != candidate["reflection"]:
        raise ValueError("Independent update used a different Reflection")
    if reflection["playbook_after_tags"] != initial:
        raise ValueError("Independent curator input already contains learned rules")
    updated_path = Path(result["updated_context_path"])
    updated_bytes = updated_path.read_bytes()
    updated_sha = hashlib.sha256(updated_bytes).hexdigest()
    if updated_sha != result["updated_context_sha256"]:
        raise ValueError("Independent updated playbook hash mismatch")
    paths[str(updated_path)] = updated_sha
    rules = added_rules(initial, updated_bytes.decode("utf-8"))
    return {"text": "\n\n".join(rule["content"] for rule in rules), "rules": rules,
            "independent_result_path": str(result_path), "independent_update_path": str(updated_path),
            "independent_update_sha256": updated_sha, "initial_playbook_sha256": digest(initial)}, paths


def build_pairs(root=ROOT, models=MODELS, content_kind="reflection"):
    """Read only pinned ADD-only selection manifests; never select or generate new candidates."""
    if content_kind not in CONTENT_KINDS:
        raise ValueError("Unknown comparison content")
    pairs, provenance, update_cache = [], {}, {}
    for model_id, _, relative in models:
        plan = root / "outputs" / relative
        if not plan.exists():
            continue
        provenance[str(plan)] = hashlib.sha256(plan.read_bytes()).hexdigest()
        groups = defaultdict(dict)
        for job in read(plan):
            condition = job["condition"]
            strategy = condition["mode"]
            if strategy not in STRATEGIES:
                continue
            directory = Path(job["path"])
            mode = job.get("sampler") or directory.parent.name
            if mode not in ("thinking", "nonthinking", "high", "low"):
                raise ValueError(f"Unknown generation mode in {directory}")
            protocol = read(directory / "protocol.json")
            if protocol["playbook_edit_mode"] != "add_only":
                raise ValueError(f"Expected ADD-only selections: {directory}")
            seed, arm = condition["seed"], condition["arm"]
            if arm not in ("without_gt", "with_gt"):
                raise ValueError("Unknown GT setting")
            split = protocol["task_orders"][str(seed)]
            train = set(split["train"])
            if train & (set(split["val"]) | set(split["test"])):
                raise ValueError("Training split overlaps evaluation splits")
            source = directory / "conditions" / condition["id"] / "selections.json"
            choices = read(source)
            provenance[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
            selected = {}
            for choice in choices:
                candidate = choice["candidate"]
                task = candidate["task_id"]
                if task not in train or task in selected:
                    raise ValueError(f"Invalid or duplicate training task: {task}")
                if digest(candidate["reflection"]) != candidate["reflection_sha256"]:
                    raise ValueError(f"Reflection hash mismatch: {task}")
                sign = 1 if strategy == "best_advantage" else -1
                if sign * choice["advantage"] < -1e-12:
                    raise ValueError("Selected Reflection violates its saved Advantage sign")
                original = Path(candidate["source_reflection_path"])
                original_bytes = original.read_bytes()
                if json.loads(original_bytes)["reflection_for_curator"] != candidate["reflection"]:
                    raise ValueError(f"Selected Reflection differs from the saved original: {task}")
                provenance[str(original)] = hashlib.sha256(original_bytes).hexdigest()
                selected[task] = {"text": candidate["reflection"], "text_sha256": candidate["reflection_sha256"],
                                  "source_repeat": candidate["source_repeat"], "source_id": candidate["source_id"],
                                  "source_path": str(original), "selection_path": str(source),
                                  "advantage": choice["advantage"], "task": candidate["task"],
                                  "mean_pct": candidate["mean_pct"]}
                if content_kind == "independent_rules":
                    if original.parents[2].resolve() != Path(protocol["source_reflection_run"]).resolve():
                        raise ValueError("Independent update belongs to a different source model/mode cohort")
                    cache_key = str(original), arm, task, candidate["source_repeat"]
                    if cache_key not in update_cache:
                        update_cache[cache_key], source_hashes = independent_rules(original, candidate, arm)
                        provenance.update(source_hashes)
                    selected[task].update(update_cache[cache_key])
                    selected[task]["reflection_sha256"] = candidate["reflection_sha256"]
                    selected[task]["text_sha256"] = digest(selected[task]["text"])
            if len(choices) != job["steps"] or set(selected) != train:
                raise ValueError(f"Incomplete selected training cohort: {source}")
            key = mode, arm, seed
            if strategy in groups[key]:
                raise ValueError(f"Duplicate selection cohort: {key}/{strategy}")
            groups[key][strategy] = selected
        for (mode, arm, seed), selections in sorted(groups.items()):
            if set(selections) != set(STRATEGIES):
                raise ValueError(f"Unpaired selection cohort: {model_id}/{mode}/{arm}/{seed}")
            best, worst = (selections[s] for s in STRATEGIES)
            if set(best) != set(worst):
                raise ValueError("Best/Weak Advantage tasks do not match")
            for task in sorted(best):
                if best[task]["task"] != worst[task]["task"]:
                    raise ValueError("Task instructions differ within a Reflection pair")
                identity = [model_id, mode, arm, seed, task, best[task]["text_sha256"], worst[task]["text_sha256"]]
                if content_kind != "reflection":
                    identity += [content_kind, best[task]["independent_update_sha256"],
                                 worst[task]["independent_update_sha256"]]
                pairs.append({"id": digest(json.dumps(identity)), "model": model_id, "mode": mode,
                              "arm": arm, "seed": seed, "task_id": task,
                              "task": best[task]["task"], "content_kind": content_kind,
                              "candidates": {s: selections[s][task] for s in STRATEGIES}})
    return pairs, provenance


class ReviewStore:
    """Durable snapshots, stable random sides, and transactional vote history for one shared reviewer."""

    def __init__(self, directory, content_kind="reflection"):
        if content_kind not in CONTENT_KINDS:
            raise ValueError("Unknown comparison content")
        self.content_kind = content_kind
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "reviews.sqlite3"
        with self.connect() as conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS pairs (
                    id TEXT PRIMARY KEY, model TEXT NOT NULL, mode TEXT NOT NULL, arm TEXT NOT NULL,
                    seed INTEGER NOT NULL, task_id TEXT NOT NULL, snapshot TEXT NOT NULL,
                    left_strategy TEXT NOT NULL, display_order TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS votes (
                    pair_id TEXT PRIMARY KEY REFERENCES pairs(id),
                    choice TEXT NOT NULL CHECK(choice IN ('left','right','tie')),
                    selected_strategy TEXT, note TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, pair_id TEXT NOT NULL,
                    choice TEXT NOT NULL, selected_strategy TEXT, note TEXT NOT NULL, at TEXT NOT NULL
                );
            """)
            conn.execute("INSERT OR IGNORE INTO settings VALUES ('shuffle_secret', ?)", (secrets.token_hex(32),))
            self.secret = conn.execute("SELECT value FROM settings WHERE key='shuffle_secret'").fetchone()[0]
            conn.execute("INSERT OR IGNORE INTO settings VALUES ('content_kind', ?)", (content_kind,))
            if conn.execute("SELECT value FROM settings WHERE key='content_kind'").fetchone()[0] != content_kind:
                raise ValueError("Use a separate database directory for rule and Reflection judgments")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.database, timeout=20)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def shuffled(self, value):
        return hmac.new(self.secret.encode(), value.encode(), hashlib.sha256).hexdigest()

    def sync(self, pairs):
        if any(pair.get("content_kind", "reflection") != self.content_kind for pair in pairs):
            raise ValueError("Cannot mix rule and Reflection review datasets")
        with self.connect() as conn:
            conn.execute("UPDATE pairs SET active=0")
            for pair in pairs:
                left = STRATEGIES[int(self.shuffled("side:" + pair["id"]), 16) % 2]
                conn.execute("""INSERT INTO pairs VALUES (?,?,?,?,?,?,?,?,?,1)
                    ON CONFLICT(id) DO UPDATE SET active=1""",
                             (pair["id"], pair["model"], pair["mode"], pair["arm"], pair["seed"],
                              pair["task_id"], json.dumps(pair, ensure_ascii=False), left,
                              self.shuffled("order:" + pair["id"])))

    def catalog(self):
        with self.connect() as conn:
            groups = [dict(row) for row in conn.execute("""
                SELECT p.model,p.mode,p.arm,p.seed,COUNT(*) AS total,COUNT(v.pair_id) AS reviewed
                FROM pairs p LEFT JOIN votes v ON p.id=v.pair_id WHERE p.active=1
                GROUP BY p.model,p.mode,p.arm,p.seed
            """)]
        return {"models": [{"id": key, "label": label,
                            "groups": [g for g in groups if g["model"] == key]} for key, label, _ in MODELS],
                "total": sum(g["total"] for g in groups), "reviewed": sum(g["reviewed"] for g in groups),
                "content_kind": self.content_kind,
                "criterion": "Which rule set is more detailed?" if self.content_kind == "independent_rules"
                else "Which Reflection is more detailed?", "csrf_token": self.shuffled("csrf")}

    def queue(self, filters):
        conditions, values = ["p.active=1"], []
        for name in ("model", "mode", "arm", "seed"):
            value = filters.get(name)
            if value and value != "all":
                conditions.append(f"p.{name}=?")
                values.append(value)
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(f"""
                SELECT p.id,p.task_id,p.seed,v.choice FROM pairs p LEFT JOIN votes v ON p.id=v.pair_id
                WHERE {' AND '.join(conditions)} ORDER BY p.display_order
            """, values)]

    def pair(self, pair_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM pairs WHERE id=? AND active=1", (pair_id,)).fetchone()
            if row is None:
                raise KeyError(pair_id)
            vote = conn.execute("SELECT choice,note,updated_at FROM votes WHERE pair_id=?", (pair_id,)).fetchone()
        snapshot = json.loads(row["snapshot"])
        right = next(s for s in STRATEGIES if s != row["left_strategy"])
        result = {"id": row["id"], "model": row["model"], "mode": row["mode"], "arm": row["arm"],
                  "seed": row["seed"], "task_id": row["task_id"], "task": snapshot["task"],
                  "left": snapshot["candidates"][row["left_strategy"]]["text"],
                  "right": snapshot["candidates"][right]["text"], "vote": dict(vote) if vote else None}
        if self.content_kind == "independent_rules":
            result.update(content_kind=self.content_kind,
                          left_rules=[r["content"] for r in snapshot["candidates"][row["left_strategy"]]["rules"]],
                          right_rules=[r["content"] for r in snapshot["candidates"][right]["rules"]])
        return result

    def vote(self, pair_id, choice, note=""):
        if choice not in ("left", "right", "tie") or not isinstance(note, str) or len(note) > 2000:
            raise ValueError("유효한 선택과 2,000자 이내 메모를 보내 주세요.")
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT left_strategy FROM pairs WHERE id=? AND active=1", (pair_id,)).fetchone()
            if row is None:
                raise KeyError(pair_id)
            selected = None if choice == "tie" else (row[0] if choice == "left" else next(s for s in STRATEGIES
                                                                                         if s != row[0]))
            old = conn.execute("SELECT choice,note FROM votes WHERE pair_id=?", (pair_id,)).fetchone()
            if old and (old["choice"], old["note"]) == (choice, note):
                return
            at = now()
            conn.execute("""INSERT INTO votes VALUES (?,?,?,?,?,?) ON CONFLICT(pair_id) DO UPDATE SET
                choice=excluded.choice,selected_strategy=excluded.selected_strategy,note=excluded.note,
                updated_at=excluded.updated_at""", (pair_id, choice, selected, note, at, at))
            conn.execute("INSERT INTO events (pair_id,choice,selected_strategy,note,at) VALUES (?,?,?,?,?)",
                         (pair_id, choice, selected, note, at))

    def export_csv(self):
        """Export visible responses only; keep source strategies blind in the browser."""
        with self.connect() as conn:
            rows = conn.execute("""SELECT p.id AS pair_id,p.model,p.mode,p.arm,p.seed,p.task_id,
                v.choice,v.note,v.created_at,v.updated_at FROM votes v JOIN pairs p ON p.id=v.pair_id
                ORDER BY v.updated_at""").fetchall()
        stream = io.StringIO()
        fields = ["pair_id", "model", "mode", "arm", "seed", "task_id", "choice", "note", "created_at", "updated_at"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            # Spreadsheet applications must not interpret reviewer notes as formulas.
            writer.writerow({k: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v
                             for k, v in dict(row).items()})
        return ("\ufeff" + stream.getvalue()).encode()


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, store, allowed_hosts=()):
        self.store = store
        self.allowed_hosts = {"localhost", "127.0.0.1", "[::1]", *allowed_hosts}
        super().__init__(address, ReviewHandler)


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewServer

    def send(self, code, body, content_type="application/json; charset=utf-8", filename=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; object-src 'none'; frame-ancestors 'none'")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def allowed(self):
        host = self.headers.get("Host", "").rsplit(":", 1)[0]
        if host not in self.server.allowed_hosts:
            self.send(HTTPStatus.FORBIDDEN, {"error": "허용되지 않은 Host입니다."})
            return False
        return True

    def do_GET(self):
        if not self.allowed():
            return
        path = urlsplit(self.path)
        if path.path == "/api/catalog":
            self.send(200, self.server.store.catalog())
        elif path.path == "/api/queue":
            filters = {key: value[-1] for key, value in parse_qs(path.query).items()}
            self.send(200, {"pairs": self.server.store.queue(filters)})
        elif path.path.startswith("/api/pairs/"):
            try:
                self.send(200, self.server.store.pair(path.path.removeprefix("/api/pairs/")))
            except KeyError:
                self.send(404, {"error": "Reflection 쌍을 찾을 수 없습니다."})
        elif path.path == "/api/export.csv":
            name = "independent_rule_reviews.csv" if self.server.store.content_kind == "independent_rules" \
                else "reflection_reviews.csv"
            self.send(200, self.server.store.export_csv(), "text/csv; charset=utf-8", name)
        elif path.path == "/favicon.ico":
            self.send(204, b"", "image/x-icon")
        else:
            files = {"/": ("index.html", "text/html; charset=utf-8"),
                     "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                     "/style.css": ("style.css", "text/css; charset=utf-8")}
            asset = files.get(path.path)
            if asset:
                self.send(200, (STATIC / asset[0]).read_bytes(), asset[1])
            else:
                self.send(404, {"error": "Not found"})

    def do_POST(self):
        if not self.allowed():
            return
        if urlsplit(self.path).path != "/api/vote":
            self.send(404, {"error": "Not found"})
            return
        origin = self.headers.get("Origin")
        if (origin and urlsplit(origin).netloc != self.headers.get("Host")) or not hmac.compare_digest(
                self.headers.get("X-Review-Token", ""), self.server.store.shuffled("csrf")):
            self.send(403, {"error": "페이지를 새로고침한 뒤 다시 선택해 주세요."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384 or self.headers.get_content_type() != "application/json":
                raise ValueError("잘못된 요청입니다.")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or not isinstance(payload.get("pair_id"), str):
                raise ValueError("Reflection 쌍을 지정해 주세요.")
            self.server.store.vote(payload["pair_id"], payload.get("choice"), payload.get("note", ""))
        except KeyError:
            self.send(404, {"error": "Reflection 쌍을 찾을 수 없습니다."})
        except (ValueError, TypeError) as exc:
            self.send(400, {"error": str(exc)})
        except sqlite3.Error:
            self.log_error("Vote transaction failed")
            self.send(503, {"error": "저장하지 못했습니다. 잠시 뒤 다시 눌러 주세요."})
        else:
            catalog = self.server.store.catalog()
            self.send(200, {"saved": True, "reviewed": catalog["reviewed"], "total": catalog["total"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8879)
    parser.add_argument("--allow-host", action="append", default=[])
    parser.add_argument("--content-kind", choices=CONTENT_KINDS, default="independent_rules")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = ROOT / "outputs/ace_reflection_detail_review_20261125" / args.content_kind
    pairs, hashes = build_pairs(content_kind=args.content_kind)
    store = ReviewStore(args.data_dir, args.content_kind)
    store.sync(pairs)
    manifest = {"created_at": now(), "pairs": len(pairs), "source_sha256": hashes,
                "criterion": "detail", "content_kind": args.content_kind,
                "pairing": "Actual saved best_advantage/worst_advantage, same task/seed/GT/mode",
                "blinding": "Persistent HMAC-shuffled sides and order; no strategies/scores in public API",
                "new_inference_calls": 0}
    (args.data_dir / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with ReviewServer((args.host, args.port), store, args.allow_host) as server:
        print(json.dumps({"url": f"http://{args.host}:{server.server_port}", "pairs": len(pairs),
                          "database": str(store.database)}, ensure_ascii=False), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
