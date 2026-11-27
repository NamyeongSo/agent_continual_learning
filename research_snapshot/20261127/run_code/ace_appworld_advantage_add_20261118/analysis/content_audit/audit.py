import csv
import json
import re
import statistics
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from scripts.ace.plot_worst_reflection_repeats import RUNS

OUT = Path(__file__).resolve().parent
read = lambda path: json.loads(path.read_text())
pairs, unique = [], {}
for mode, run in RUNS.items():
    for arm in ("without_gt", "with_gt"):
        source = Path(read(run / "protocol.json")["source_reflection_run"])
        for seed in (42, 43, 44):
            selections = {}
            for kind in ("best", "worst"):
                folder = run / "conditions" / f"s{seed}_{arm}_{kind}"
                rows = read(folder / "selections.json")
                for i, row in enumerate(rows, 1):
                    c = row["candidate"]
                    after = (folder / "checkpoints" / f"{i:03}.txt").read_text().splitlines()
                    before = set((folder / "checkpoints" / f"{i-1:03}.txt").read_text().splitlines())
                    delta = [line.split(" :: ", 1)[1] for line in after if " :: " in line and line not in before]
                    proposal = read(source / "stage2" / f'{arm}_{c["source_id"]}' / "result.json")
                    independent = Path(proposal["updated_context_path"]).read_text()
                    selections.setdefault(c["task_id"], {})[kind] = {
                        "candidate": c["source_id"], "reflection": c["reflection"],
                        "new_rules": delta, "independent_rules": [
                            line.split(" :: ", 1)[1] for line in independent.splitlines() if " :: " in line
                        ], "step": i, "reward": [c["score_numerator"], c["score_denominator"]],
                    }
            assert len(selections) == 50
            for task, data in selections.items():
                b, w = data["best"], data["worst"]
                key = (mode, arm, task, b["candidate"], w["candidate"])
                pair = {"mode": mode, "arm": arm, "seed": seed, "task": task, **data}
                pair["tie"] = b["reward"] == w["reward"]
                pair["same_candidate"] = b["candidate"] == w["candidate"]
                pairs.append(pair)
                unique.setdefault(key, pair)
assert len(pairs) == 600
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu", local_files_only=True)
import torch
torch.set_num_threads(4)

def segments(text):
    return [s.strip() for s in re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z])", text) if len(s.strip()) >= 15] or [text]

def texts(pair, side, field):
    value = pair[side][field]
    return segments(value) if field == "reflection" else value

fields = ("reflection", "independent_rules", "new_rules")
items = list(dict.fromkeys(
    text for p in pairs for side in ("best", "worst") for field in fields for text in texts(p, side, field)
))
emb = dict(zip(items, model.encode(items, normalize_embeddings=True, batch_size=64, show_progress_bar=False)))
for pair in pairs:
    for field in fields:
        b, w = (texts(pair, side, field) for side in ("best", "worst"))
        if not b or not w:
            pair[field + "_similarity"] = None
            continue
        matrix = np.stack([emb[s] for s in b]) @ np.stack([emb[s] for s in w]).T
        pair[field + "_similarity"] = float((matrix.max(axis=0).mean() + matrix.max(axis=1).mean()) / 2)
        if field != "reflection":
            pair[field + "_most_distinct_best"] = b[int(matrix.max(axis=1).argmin())]
            pair[field + "_most_distinct_worst"] = w[int(matrix.max(axis=0).argmin())]

def aggregate(rows):
    d = {"pairs": len(rows), "score_ties": sum(p["tie"] for p in rows),
         "same_candidate": sum(p["same_candidate"] for p in rows),
         "both_added_rules": sum(bool(p["best"]["new_rules"] and p["worst"]["new_rules"]) for p in rows),
         "best_only_added_rules": sum(bool(p["best"]["new_rules"] and not p["worst"]["new_rules"]) for p in rows),
         "worst_only_added_rules": sum(bool(p["worst"]["new_rules"] and not p["best"]["new_rules"]) for p in rows),
         "neither_added_rules": sum(not p["worst"]["new_rules"] and not p["best"]["new_rules"] for p in rows)}
    for field in fields:
        vals = [p[field + "_similarity"] for p in rows if p[field + "_similarity"] is not None]
        d[field] = {"n": len(vals), "median": statistics.median(vals) if vals else None,
                     "ge_0.8": sum(v >= .8 for v in vals), "lt_0.6": sum(v < .6 for v in vals)}
    return d

u = list(unique.values())
summary = {"all_training_pairs": aggregate(pairs), "unique_candidate_pairs": aggregate(u),
           "positive_gap_unique": aggregate([p for p in u if not p["tie"]]),
           "positive_gap_training_pairs": aggregate([p for p in pairs if not p["tie"]]),
           "positive_gap_by_condition": {
               f"{mode}/{arm}": aggregate([p for p in u if not p["tie"] and (p["mode"], p["arm"]) == (mode, arm)])
               for mode in RUNS for arm in ("without_gt", "with_gt")}}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
(OUT / "all_pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2))
with (OUT / "pairs.csv").open("w") as f:
    columns = ["mode", "arm", "seed", "task", "tie", "same_candidate", "best_id", "worst_id",
               *[field + "_similarity" for field in fields], "best_new_rules", "worst_new_rules"]
    w = csv.DictWriter(f, fieldnames=columns)
    w.writeheader()
    for p in pairs:
        w.writerow({**{k: p[k] for k in columns if k in p}, "best_id": p["best"]["candidate"],
                    "worst_id": p["worst"]["candidate"], "best_new_rules": len(p["best"]["new_rules"]),
                    "worst_new_rules": len(p["worst"]["new_rules"])})
print(json.dumps(summary, indent=2))
