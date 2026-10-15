"""Build single-file edit tasks from public git history.

A task is one real commit that changes exactly one non-test Python file with
1-3 hunks and 1-20 changed lines. We keep the file before and after the commit.
No model is involved here. Selection is deterministic (seeded).

usage: python build_tasks.py <repos_dir> <out.jsonl> <per_repo> <repo> [<repo> ...]
"""
import json, random, re, subprocess, sys
from pathlib import Path

def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace").stdout

def is_test(path):
    p = path.lower()
    return ("test" in Path(p).name) or "/tests/" in "/" + p or p.startswith("tests/") \
        or "/testing/" in p or p.startswith("docs") or "docs_src" in p or "/examples/" in "/" + p

def hunks(diff):
    return [h for h in re.split(r"(?m)^@@", diff)[1:]]

def changed_lines(diff):
    out = []
    for line in diff.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            out.append(line[1:])
    return out

def meaningful(lines):
    for l in lines:
        s = l.strip()
        if s and not s.startswith("#"):
            return True
    return False

def candidates(repo):
    log = git(repo, "log", "--no-merges", "--format=@@C %H %cI %s", "--numstat")
    cur = None
    for line in log.splitlines():
        if line.startswith("@@C "):
            if cur:
                yield cur
            _, sha, date, *subj = line.split(" ")
            cur = {"sha": sha, "date": date, "subject": " ".join(subj), "files": []}
        elif line.strip() and cur is not None:
            parts = line.split("\t")
            if len(parts) == 3:
                cur["files"].append(parts)
    if cur:
        yield cur

def build(repo, per_repo, rng):
    pool = []
    for c in candidates(repo):
        py = [f for f in c["files"] if f[2].endswith(".py") and not is_test(f[2])]
        if len(py) != 1:
            continue
        add, rem, path = py[0]
        if add == "-" or not (1 <= int(add) + int(rem) <= 20):
            continue
        pool.append((c, path))
    rng.shuffle(pool)
    tasks = []
    for c, path in pool:
        if len(tasks) >= per_repo:
            break
        diff = git(repo, "diff", "-U0", c["sha"] + "^", c["sha"], "--", path)
        hs = hunks(diff)
        if not (1 <= len(hs) <= 3) or not meaningful(changed_lines(diff)):
            continue
        before = git(repo, "show", f"{c['sha']}^:{path}")
        after = git(repo, "show", f"{c['sha']}:{path}")
        n = before.count("\n")
        if not (30 <= n <= 800) or before == after or "\t" in before[:0]:
            continue
        tasks.append({
            "id": f"{Path(repo).name}_{c['sha'][:10]}",
            "repo": Path(repo).name, "sha": c["sha"], "date": c["date"],
            "subject": c["subject"], "path": path, "n_hunks": len(hs),
            "n_lines_before": n, "before": before, "after": after,
        })
    return tasks

def main():
    repos_dir, out, per_repo, *repos = sys.argv[1:]
    rng = random.Random(20261015)
    allt = []
    for r in repos:
        t = build(Path(repos_dir) / r, int(per_repo), rng)
        print(r, len(t), file=sys.stderr)
        allt += t
    with open(out, "w", encoding="utf-8") as f:
        for t in allt:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
