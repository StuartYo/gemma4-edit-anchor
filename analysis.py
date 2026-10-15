"""Paired comparisons between conditions (same tasks, same model).

usage: python analysis.py --tasks data/dev.jsonl data/holdout.jsonl --out results/paired.json
Reads runs/*_target_*_s1*.jsonl, merges _pN slices, drops whitespace-only tasks.
For each model/split, compares every condition with S on per-task solved (exact McNemar test)
and on "at least one edit applied".
"""
import argparse, glob, json, math, re
from collections import defaultdict
from run_exp import norm

def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value; b = only A solved, c = only B solved."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)

def load(tasks):
    runs = defaultdict(dict)
    for f in glob.glob("runs/*_target_*_s1*.jsonl"):
        name = re.sub(r"_p\d+$", "", f.replace("\\", "/").split("/")[-1].replace(".jsonl", ""))
        for line in open(f, encoding="utf-8"):
            l = json.loads(line)
            t = tasks[l["id"]]
            if norm(t["before"]) == norm(t["after"]):
                continue
            runs[name][l["id"]] = l
    return runs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+"); ap.add_argument("--out")
    a = ap.parse_args()
    tasks = {}
    for tf in a.tasks:
        for line in open(tf, encoding="utf-8"):
            t = json.loads(line); tasks[t["id"]] = t
    runs = load(tasks)
    out = {}
    for name, logs in sorted(runs.items()):
        m = re.match(r"(E\dB)_(\w+?)_target_(\w+)_s1$", name)
        if not m or m.group(3) == "S":
            continue
        model, split, cond = m.groups()
        base = runs.get(f"{model}_{split}_target_S_s1")
        if not base:
            continue
        ids = sorted(set(base) & set(logs))
        def solved(l): return bool(l.get("solved"))
        def applied(l): return any(c.get("applied") for c in l["calls"])
        res = {"n_paired": len(ids)}
        for metric, fn in [("solved", solved), ("any_edit_applied", applied)]:
            b = sum(1 for i in ids if fn(base[i]) and not fn(logs[i]))
            c = sum(1 for i in ids if fn(logs[i]) and not fn(base[i]))
            res[metric] = {"S": sum(fn(base[i]) for i in ids), cond: sum(fn(logs[i]) for i in ids),
                           "only_S": b, f"only_{cond}": c, "mcnemar_p": round(mcnemar_exact(b, c), 4)}
        out[f"{model}_{split}_{cond}_vs_S"] = res
    s = json.dumps(out, indent=1)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(s)
    print(s)

if __name__ == "__main__":
    main()
