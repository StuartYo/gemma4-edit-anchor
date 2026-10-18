"""Figures for the paper, drawn only from results/summary.json.

usage: python make_figs.py results/summary.json figures/
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = json.load(open(sys.argv[1], encoding="utf-8"))
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c9c8c2",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})

def key(model, split, mode, cond, seed=1):
    return f"{model}_{split}_{mode}_{cond}_s{seed}"

# Fig 1: solved rate by condition, per model and split (dashed line = S of that model)
def fig1():
    conds = [("S", "S: strict\n(harness)"), ("N", "N: strict +\nnearest block"),
             ("R", "R: silent\nrepair"), ("RN", "RN: repair +\nnearest block"),
             ("L", "L: line range +\nfirst-line check")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, gridspec_kw={"width_ratios": [5, 3]})
    for ax, split, title in [(axes[0], "dev100", "Development repositories (98 tasks)"),
                             (axes[1], "holdout60", "Held-out repositories (60 tasks)")]:
        present = [(c, lab) for c, lab in conds if any(S.get(key(m, split, "target", c)) for m in ("E2B", "E4B"))]
        for mi, (model, color) in enumerate([("E2B", BLUE), ("E4B", ORANGE)]):
            xs, ys, lo, hi = [], [], [], []
            for ci, (c, _) in enumerate(present):
                r = S.get(key(model, split, "target", c))
                if not r:
                    continue
                xs.append(ci + (mi - 0.5) * 0.36)
                ys.append(r["solved_rate"] * 100)
                lo.append((r["solved_rate"] - r["solved_ci95"][0]) * 100)
                hi.append((r["solved_ci95"][1] - r["solved_rate"]) * 100)
            if xs:
                ax.bar(xs, ys, width=0.34, color=color, label=f"Gemma 4 {model}", edgecolor="white", linewidth=2)
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
                for x, y in zip(xs, ys):
                    ax.text(x, 1.5, f"{y:.0f}", ha="center", va="bottom", fontsize=8, color="white", weight="bold")
            base = S.get(key(model, split, "target", "S"))
            if base:
                ax.axhline(base["solved_rate"] * 100, color=color, lw=0.8, ls="--", alpha=0.6)
        ax.set_xticks(range(len(present))); ax.set_xticklabels([l for _, l in present], fontsize=7.5)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.yaxis.grid(True, color="#ecebe6"); ax.set_axisbelow(True)
    axes[0].set_ylabel("Tasks solved exactly (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "fig1_solved_by_condition.png", dpi=200); plt.close(fig)

# Fig 2: why strict edit calls fail (all rejected calls, condition S, development, target mode)
LABELS = {"near_miss": "Near miss (similarity >= 0.9)", "partial_paraphrase": "Partly rewritten (0.6-0.9)",
          "quoted_the_target": "Quoted the edited version", "line_numbers_copied": "Copied line numbers",
          "whitespace_or_indent": "Whitespace / indentation only", "not_unique": "Matches more than once",
          "not_in_file": "Not in file (< 0.6)", "empty_old": "Empty old_string"}
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), sharex=True, sharey=True)
    for ax, model, color in [(axes[0], "E2B", BLUE), (axes[1], "E4B", ORANGE)]:
        r = S.get(key(model, "dev100", "target", "S"))
        if not r:
            continue
        tax = r["all_fail_taxonomy"]; tot = sum(tax.values()) or 1
        items = [(LABELS[k], tax.get(k, 0) * 100 / tot) for k in reversed(list(LABELS))]  # same order in both panels
        ax.barh([a for a, _ in items], [b for _, b in items], color=color, height=0.6)
        for i, (_, b) in enumerate(items):
            ax.text(b + 1, i, f"{b:.0f}%", va="center", fontsize=8, color=INK)
        ax.set_title(f"Gemma 4 {model}: {tot} rejected calls", fontsize=10, color=INK, loc="left")
        ax.xaxis.grid(True, color="#ecebe6"); ax.set_axisbelow(True)
    axes[0].set_xlabel("Share of rejected calls (%)"); axes[1].set_xlabel("Share of rejected calls (%)")
    fig.tight_layout(); fig.savefig(OUT / "fig2_failure_taxonomy.png", dpi=200); plt.close(fig)

# Fig 3: replay of first calls, per model (S runs, both splits). Appliers are nested:
# lineno includes strict, ws includes lineno, fuzzy includes ws.
def fig3():
    order = [("lineno", "+ strip line\nnumbers"), ("ws", "+ whitespace / indent\n(unique only; = R)"),
             ("fuzzy", "+ difflib >= 0.8\n(best block)")]
    fields = [("repaired_right_place", "Recovered, overlaps gold lines", BLUE),
              ("repaired_wrong_place", "Recovered, wrong place", ORANGE),
              ("repaired_moved_away_from_gold", "Of all recovered: file moved away from gold", GRAY)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    for ax, model in zip(axes, ("E2B", "E4B")):
        agg = {}
        for split in ("dev100", "holdout60"):
            r = S.get(key(model, split, "target", "S"))
            if not r:
                continue
            for ap, v in r["replay_first_call"].items():
                a = agg.setdefault(ap, {})
                for f, x in v.items():
                    a[f] = a.get(f, 0) + x
        rej = agg.get("strict", {}).get("rejected", 0)
        for fi, (f, lab, color) in enumerate(fields):
            xs = [i + (fi - 1) * 0.27 for i in range(len(order))]
            ys = [agg.get(ap, {}).get(f, 0) for ap, _ in order]
            ax.bar(xs, ys, width=0.25, color=color, label=lab, edgecolor="white", linewidth=1.5)
            for x, y in zip(xs, ys):
                ax.text(x, y + 0.5, str(y), ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([l for _, l in order], fontsize=8)
        ax.set_title(f"Gemma 4 {model}: {rej} first calls rejected by strict", fontsize=10, color=INK, loc="left")
        ax.yaxis.grid(True, color="#ecebe6"); ax.set_axisbelow(True)
    axes[0].set_ylabel("Rejected first calls")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig3_repair_tradeoff.png", dpi=200); plt.close(fig)

fig1(); fig2(); fig3()
print("figures written to", OUT)
