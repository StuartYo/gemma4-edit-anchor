"""Score run logs. All numbers in the paper come from this script.

usage: python score.py runs/*.jsonl --tasks data/dev.jsonl data/holdout.jsonl --out results/summary.json
"""
import argparse, glob, json, random
from collections import Counter, defaultdict
import appliers as A

def first_edit(log):
    for c in log["calls"]:
        if c["name"] == "edit_file" and "old" in c:
            return c
    return None

def overlaps(span, gold, pad=3):
    s, e = span
    return any(s < ge + pad and gs - pad < e for gs, ge in gold)

def classify(text, old, new):
    """Why did a strict first call fail?"""
    r, info = A.strict(text, old, new)
    if r is not None:
        return "ok"
    if info.get("error") == "multiple":
        return "not_unique"
    if old.strip() == "":
        return "empty_old"
    if A.lineno(text, old, new)[0] is not None:
        return "line_numbers_copied"
    r, info = A.ws(text, old, new)
    if r is not None:
        return "whitespace_or_indent"
    if info.get("error") == "ws_multiple":
        return "not_unique"
    _, _, ratio = A.nearest_block(text, old)
    if ratio >= 0.9:
        return "near_miss"
    if ratio >= 0.6:
        return "partial_paraphrase"
    return "not_in_file"

def boot_ci(xs, n=2000, seed=0):
    if not xs:
        return (None, None)
    rng = random.Random(seed)
    ms = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return (round(ms[int(0.025 * n)], 3), round(ms[int(0.975 * n)], 3))

def summarize(logs, tasks):
    n = len(logs)
    out = {"n": n}
    fe = [first_edit(l) for l in logs]
    has = [f for f in fe if f is not None]
    out["no_edit_call"] = sum(f is None for f in fe)
    out["malformed_calls"] = sum(1 for l in logs for c in l["calls"] if c.get("malformed"))
    out["first_strict_ok"] = sum(1 for l, f in zip(logs, fe) if f and A.strict(tasks[l["id"]]["before"], f["old"], f["new"])[0] is not None)
    # taxonomy of first-call failures (state = original file, first edit is always first)
    tax = Counter()
    replay = {k: Counter() for k in A.APPLIERS}
    for l, f in zip(logs, fe):
        if not f:
            continue
        t = tasks[l["id"]]
        text = t["before"]
        cls = classify(text, f["old"], f["new"])
        if cls != "ok":
            tax[cls] += 1
        for name, fn in A.APPLIERS.items():
            r, info = fn(text, f["old"], f["new"])
            if r is None:
                replay[name]["rejected"] += 1
            else:
                ok_loc = overlaps(info["span"], l["gold_spans"])
                replay[name]["applied"] += 1
                replay[name]["applied_wrong_place" if not ok_loc else "applied_right_place"] += 1
                if info.get("how") != "exact":
                    replay[name]["repaired"] += 1
                    replay[name]["repaired_wrong_place" if not ok_loc else "repaired_right_place"] += 1
    out["first_fail_taxonomy"] = dict(tax)
    # taxonomy over every failed edit call (state saved at call time)
    allt = Counter()
    for l in logs:
        for c in l["calls"]:
            if c["name"] == "edit_file" and "old" in c and not c.get("applied") and c.get("before_call"):
                allt[classify(c["before_call"], c["old"], c["new"])] += 1
    out["all_fail_taxonomy"] = dict(allt)
    out["arg_parse_errors"] = sum(l.get("parse_errors", 0) for l in logs)
    out["tasks_with_parse_error"] = sum(1 for l in logs if l.get("parse_errors"))
    out["replay_first_call"] = {k: dict(v) for k, v in replay.items()}
    if logs and logs[0]["mode"] == "target":
        solved = [1 if l.get("solved") else 0 for l in logs]
        out["solved"] = sum(solved)
        out["solved_rate"] = round(sum(solved) / n, 3)
        out["solved_ci95"] = boot_ci(solved)
    out["changed"] = sum(1 for l in logs if l["changed"])
    out["edit_calls"] = sum(1 for l in logs for c in l["calls"] if c["name"] == "edit_file")
    out["failed_edit_calls"] = sum(1 for l in logs for c in l["calls"] if c["name"] == "edit_file" and not c.get("applied"))
    out["mean_secs"] = round(sum(l["secs"] for l in logs) / max(1, n), 1)
    out["mean_completion_tokens"] = round(sum(l["completion_tokens"] for l in logs) / max(1, n), 1)
    out["server_errors"] = sum(l.get("server_errors", 0) for l in logs)
    out["stopped_without_edit_first"] = sum(1 for l in logs if l.get("stopped_without_edit_first"))
    out["nudged"] = sum(1 for l in logs if l.get("nudges"))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+"); ap.add_argument("--tasks", nargs="+"); ap.add_argument("--out")
    a = ap.parse_args()
    tasks = {}
    for tf in a.tasks:
        for line in open(tf, encoding="utf-8"):
            t = json.loads(line); tasks[t["id"]] = t
    res = {}
    files = sorted({f for p in a.runs for f in glob.glob(p)})
    for f in files:
        logs = [json.loads(l) for l in open(f, encoding="utf-8")]
        res[f.replace("\\", "/").split("/")[-1].replace(".jsonl", "")] = summarize(logs, tasks)
    s = json.dumps(res, indent=1, ensure_ascii=False)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(s)
    print(s)

if __name__ == "__main__":
    main()
