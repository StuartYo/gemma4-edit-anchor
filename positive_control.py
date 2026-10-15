"""Positive control: correct old_strings built from the gold diff must apply and
reproduce the gold file under every applier. Validates tasks + appliers."""
import difflib, json, sys
import appliers as A
def gold_edits(before, after):
    b, a = before.split("\n"), after.split("\n")
    ops = [o for o in difflib.SequenceMatcher(None, b, a, autojunk=False).get_opcodes() if o[0] != "equal"]
    edits = []
    for tag, i1, i2, j1, j2 in ops:
        c = 1
        while True:
            lo, hi = max(0, i1 - c), min(len(b), i2 + c)
            old = "\n".join(b[lo:hi]); 
            if before.count(old) == 1 or c > 30: break
            c += 1
        new = "\n".join(b[lo:i1] + a[j1:j2] + b[i2:hi])
        edits.append((old, new))
    return edits
bad = {k: 0 for k in A.APPLIERS}; n = 0
for f in sys.argv[1:]:
    for line in open(f, encoding="utf-8"):
        t = json.loads(line); n += 1
        for name, fn in A.APPLIERS.items():
            text = t["before"]; ok = True
            for old, new in gold_edits(t["before"], t["after"]):
                r, _ = fn(text, old, new)
                if r is None: ok = False; break
                text = r
            if not ok or text.rstrip() != t["after"].rstrip(): bad[name] += 1
print("tasks", n, "failures per applier", bad)

# Line-range interface (condition L): gold edits as (start, end, first_line, new), applied bottom-up
from run_exp import apply_lines
badL = 0
for f in sys.argv[1:]:
    for line in open(f, encoding="utf-8"):
        t = json.loads(line)
        b, a = t["before"].splitlines(), t["after"].splitlines()
        ops = [o for o in difflib.SequenceMatcher(None, b, a, autojunk=False).get_opcodes() if o[0] != "equal"]
        text = t["before"]
        for tag, i1, i2, j1, j2 in reversed(ops):
            start, end = i1 + 1, i2
            first = b[i1] if i1 < len(b) else ""
            # convention: every line of new_content ends with a newline
            new = "".join(x + "\n" for x in a[j1:j2])
            r, info = apply_lines(text, start, end, first, new)
            if r is None:
                badL += 1; break
            text = r
        else:
            if text.rstrip() != t["after"].rstrip():
                badL += 1
print("line-range interface failures", badL)
