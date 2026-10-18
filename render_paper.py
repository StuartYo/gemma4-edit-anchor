"""Fill {{ expression }} placeholders in the paper template from the result files,
so no number in the paper is typed by hand.

usage: python render_paper.py paper_template.md results/summary.json results/paired.json paper.md
Inside {{ }}: S = summary dict, P = paired dict, and helpers pct(a, b), g(run, key).
"""
import json, re, sys

tpl, sp, pp, outp = sys.argv[1:5]
S = json.load(open(sp, encoding="utf-8"))
P = json.load(open(pp, encoding="utf-8"))

def pct(a, b, nd=0):
    v = 100.0 * a / b if b else 0.0
    return f"{v:.{nd}f}%"

def g(run, *keys):
    v = S[run]
    for k in keys:
        v = v.get(k, 0) if isinstance(v, dict) else 0
    return v

def ci(run):
    lo, hi = S[run]["solved_ci95"]
    return f"{lo * 100:.0f} to {hi * 100:.0f}%"

env = {"S": S, "P": P, "pct": pct, "g": g, "ci": ci}
text = open(tpl, encoding="utf-8").read()
missing = []

def sub(m):
    try:
        return str(eval(m.group(1), env))
    except Exception as e:
        missing.append(f"{m.group(1)} -> {e!r}")
        return "[[MISSING]]"

out = re.sub(r"\{\{(.+?)\}\}", sub, text)
open(outp, "w", encoding="utf-8").write(out)
print("rendered", outp, "missing:", len(missing))
for x in missing:
    print("  ", x)
