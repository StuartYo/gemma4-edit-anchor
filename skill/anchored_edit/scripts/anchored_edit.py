"""Standalone anchored edit (no dependencies). Mirrors appliers.ws + nearest_block."""
import argparse, difflib, re, sys
LINENO = re.compile(r"^\s*\d+(\t|: |\| | \| |→|  )")
def strip_lineno(s):
    lines = s.split("\n"); ne = [l for l in lines if l.strip()]
    if ne and all(LINENO.match(l) for l in ne):
        return "\n".join(LINENO.sub("", l, count=1) for l in lines), True
    return s, False
def ind(l): return len(l) - len(l.lstrip(" "))
def apply(text, old, new):
    if old and text.count(old) == 1:
        return text.replace(old, new, 1), "exact"
    o, did = strip_lineno(old); n, _ = strip_lineno(new)
    if did and o and text.count(o) == 1:
        return text.replace(o, n, 1), "lineno"
    fl = text.split("\n"); ol = o.strip("\n").split("\n")
    if not any(l.strip() for l in ol): return None, "empty"
    key = [l.strip() for l in ol]; hits = []
    for s in range(len(fl) - len(ol) + 1):
        w = fl[s:s + len(ol)]
        if [l.strip() for l in w] != key: continue
        sh = {ind(a) - ind(b) for a, b in zip(w, ol) if b.strip()}
        if len(sh) == 1: hits.append((s, sh.pop()))
    if len(hits) != 1: return None, f"{len(hits)} matches"
    s, sh = hits[0]
    nl = n.strip("\n").split("\n") if n.strip("\n") else []
    rep = [("" if not l.strip() else (" " * sh + l if sh >= 0 else l[min(-sh, ind(l)):])) for l in nl]
    return "\n".join(fl[:s] + rep + fl[s + len(ol):]), f"indent shift {sh}"
def nearest(text, old):
    o, _ = strip_lineno(old); fl = text.split("\n"); ol = o.strip("\n").split("\n"); k = max(1, len(ol))
    t = "\n".join(l.strip() for l in ol); best, bs = 0.0, 0
    for s in range(max(1, len(fl) - k + 1)):
        sm = difflib.SequenceMatcher(None, t, "\n".join(l.strip() for l in fl[s:s + k]), autojunk=False)
        if sm.quick_ratio() < best: continue
        r = sm.ratio()
        if r > best: best, bs = r, s
    return "\n".join(f"{bs + 1 + i:6d}\t{l}" for i, l in enumerate(fl[bs:bs + k]))
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--file"); ap.add_argument("--old"); ap.add_argument("--new")
    a = ap.parse_args()
    text = open(a.file, encoding="utf-8").read()
    old = open(a.old, encoding="utf-8").read(); new = open(a.new, encoding="utf-8").read()
    res, how = apply(text, old, new)
    if res is None:
        print(f"NOT APPLIED ({how}). Closest block in {a.file}; copy it without the numbers:\n" + nearest(text, old)); sys.exit(1)
    open(a.file, "w", encoding="utf-8").write(res); print(f"APPLIED ({how})")
