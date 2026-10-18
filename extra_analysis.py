"""Secondary analyses for the paper (no GPU): per-repository solve rates, commits before
vs from 2025, seed-to-seed paired agreement, and condition L details.

usage: python extra_analysis.py --tasks data/dev.jsonl data/holdout.jsonl --out results/extra.json
Reads runs/*.jsonl, merges _pN slices, drops whitespace-only tasks (same rules as score.py).
"""
import argparse, glob, json, math, re
from collections import Counter, defaultdict
from run_exp import norm
from analysis import mcnemar_exact

def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact p for [[a, b], [c, d]]."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    def p(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    p0 = p(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9)))

def load(tasks):
    runs = defaultdict(dict)
    for f in glob.glob("runs/*.jsonl"):
        name = re.sub(r"_p\d+$", "", f.replace("\\", "/").split("/")[-1].replace(".jsonl", ""))
        for line in open(f, encoding="utf-8"):
            l = json.loads(line)
            t = tasks[l["id"]]
            if norm(t["before"]) == norm(t["after"]):
                continue
            runs[name][l["id"]] = l
    return runs

def first_call_ok(l):
    eds = [c for c in l["calls"] if c["name"] in ("edit_file", "edit_lines") and ("old" in c or "lines" in c)]
    return bool(eds) and bool(eds[0].get("applied"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+"); ap.add_argument("--out")
    a = ap.parse_args()
    tasks = {}
    for tf in a.tasks:
        for line in open(tf, encoding="utf-8"):
            t = json.loads(line); tasks[t["id"]] = t
    runs = load(tasks)
    out = {"per_repo": {}, "by_year": {}, "seed_pairs": {}, "L": {}}

    # per-repository solved, target mode, seed 1
    for name, logs in sorted(runs.items()):
        m = re.match(r"(E\dB)_(\w+?)_target_(\w+)_s1$", name)
        if not m:
            continue
        per = defaultdict(lambda: [0, 0])
        for l in logs.values():
            per[l["repo"]][1] += 1
            per[l["repo"]][0] += bool(l.get("solved"))
        out["per_repo"][name] = {r: {"solved": s, "n": n} for r, (s, n) in sorted(per.items())}

    # commits before 2025 vs from 2025 on, condition S, seed 1, dev + holdout pooled per model
    for model in ("E2B", "E4B"):
        cell = {"pre2025": Counter(), "from2025": Counter()}
        for split in ("dev100", "holdout60"):
            for l in runs.get(f"{model}_{split}_target_S_s1", {}).values():
                k = "from2025" if tasks[l["id"]]["date"][:4] >= "2025" else "pre2025"
                cell[k]["n"] += 1
                cell[k]["solved"] += bool(l.get("solved"))
                cell[k]["first_call_ok"] += first_call_ok(l)
        res = {k: dict(v) for k, v in cell.items()}
        for metric in ("solved", "first_call_ok"):
            x1, n1 = cell["pre2025"][metric], cell["pre2025"]["n"]
            x2, n2 = cell["from2025"][metric], cell["from2025"]["n"]
            res[f"fisher_p_{metric}"] = round(fisher_exact(x1, n1 - x1, x2, n2 - x2), 4) if n1 and n2 else None
        out["by_year"][model] = res

    # seed 1 vs seed 2, paired on the same tasks
    for cond in ("S", "L"):
        r1, r2 = runs.get(f"E2B_dev100_target_{cond}_s1"), runs.get(f"E2B_dev100_target_{cond}_s2")
        if not (r1 and r2):
            continue
        ids = sorted(set(r1) & set(r2))
        res = {"n_paired": len(ids)}
        for metric, fn in [("solved", lambda l: bool(l.get("solved"))),
                           ("any_edit_applied", lambda l: any(c.get("applied") for c in l["calls"])),
                           ("first_call_ok", first_call_ok)]:
            b = sum(1 for i in ids if fn(r1[i]) and not fn(r2[i]))
            c = sum(1 for i in ids if fn(r2[i]) and not fn(r1[i]))
            res[metric] = {"s1": sum(fn(r1[i]) for i in ids), "s2": sum(fn(r2[i]) for i in ids),
                           "only_s1": b, "only_s2": c, "mcnemar_p": round(mcnemar_exact(b, c), 4)}
        out["seed_pairs"][f"E2B_dev100_{cond}"] = res

    # condition L: how edit_lines calls fail
    for name, logs in sorted(runs.items()):
        if "_L_" not in name:
            continue
        calls = [c for l in logs.values() for c in l["calls"] if c["name"] == "edit_lines" and "lines" in c]
        out["L"][name] = {"edit_lines_calls": len(calls),
                          "applied": sum(1 for c in calls if c.get("applied")),
                          "errors": dict(Counter(c["info"].get("error") for c in calls if not c.get("applied"))),
                          "other_tool_calls": sum(1 for l in logs.values() for c in l["calls"] if c["name"] not in ("edit_lines", "read_file"))}
    # paired difference in solved tasks (condition minus S): 95% bootstrap CI over tasks, and the
    # smallest difference an exact McNemar test would detect with ~80% power at these discordant counts
    out["paired_diff"] = {}
    rng = __import__("random").Random(0)
    for name, logs in sorted(runs.items()):
        m = re.match(r"(E\dB)_(\w+?)_target_(\w+)_s1$", name)
        if not m or m.group(3) == "S":
            continue
        base = runs.get(f"{m.group(1)}_{m.group(2)}_target_S_s1")
        ids = sorted(set(base) & set(logs))
        d = [int(bool(logs[i].get("solved"))) - int(bool(base[i].get("solved"))) for i in ids]
        boots = sorted(sum(rng.choice(d) for _ in d) for _ in range(2000))
        disc = sum(1 for x in d if x)
        out["paired_diff"][name] = {"n": len(d), "diff": sum(d), "ci95": [boots[50], boots[1949]],
                                    "discordant": disc, "mde80_tasks": round(2.8 * math.sqrt(disc), 1)}

    # among episodes where an edit applied but the task was not solved: how often does the final
    # file have a different number of lines than the gold file (lines dropped or duplicated)?
    out["applied_not_solved_line_count"] = {}
    for name, logs in sorted(runs.items()):
        if not re.match(r"E\dB_\w+?_target_(S|L)_s1$", name):
            continue
        ans = [l for l in logs.values() if not l.get("solved") and any(c.get("applied") for c in l["calls"])]
        diff = sum(1 for l in ans if len(norm(l["final"]).split("\n")) != len(norm(tasks[l["id"]]["after"]).split("\n")))
        out["applied_not_solved_line_count"][name] = {"applied_not_solved": len(ans), "line_count_differs": diff}
    s = json.dumps(out, indent=1)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(s)
    print(s)

if __name__ == "__main__":
    main()
