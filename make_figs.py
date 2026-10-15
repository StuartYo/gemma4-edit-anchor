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

# Fig 1: solved rate by condition, per model and split
def fig1():
    conds = [("S", "S: strict\n(harness)"), ("N", "N: strict +\nnearest block"),
             ("R", "R: silent\nrepair"), ("RN", "RN: repair +\nnearest block")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, split, title in [(axes[0], "dev100", "Development (100 tasks)"), (axes[1], "holdout60", "Held-out repos (60 tasks)")]:
        for mi, (model, color) in enumerate([("E2B", BLUE), ("E4B", ORANGE)]):
            xs, ys, lo, hi = [], [], [], []
            for ci, (c, _) in enumerate(conds):
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
                    ax.text(x, y + 1.5, f"{y:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in conds], fontsize=8)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.yaxis.grid(True, color="#ecebe6"); ax.set_axisbelow(True)
    axes[0].set_ylabel("Tasks solved exactly (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig1_solved_by_condition.png", dpi=200); plt.close(fig)

# Fig 2: why strict edit calls fail (all failed calls, condition S, dev)
LABELS = {"line_numbers_copied": "Copied line numbers", "whitespace_or_indent": "Whitespace / indentation only",
          "near_miss": "Near miss (similarity >= 0.9)", "partial_paraphrase": "Partly rewritten (0.6-0.9)",
          "not_in_file": "Not in file (< 0.6)", "not_unique": "Matches more than once", "empty_old": "Empty old_string"}
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), sharex=True)
    for ax, model, color in [(axes[0], "E2B", BLUE), (axes[1], "E4B", ORANGE)]:
        r = S.get(key(model, "dev100", "target", "S"))
        if not r:
            continue
        tax = r["all_fail_taxonomy"]; tot = sum(tax.values()) or 1
        items = [(LABELS[k], tax.get(k, 0) * 100 / tot) for k in LABELS]
        items.sort(key=lambda x: x[1])
        ax.barh([a for a, _ in items], [b for _, b in items], color=color, height=0.6)
        for i, (_, b) in enumerate(items):
            ax.text(b + 1, i, f"{b:.0f}%", va="center", fontsize=8, color=INK)
        ax.set_title(f"Gemma 4 {model}: {tot} rejected edit_file calls", fontsize=10, color=INK, loc="left")
        ax.xaxis.grid(True, color="#ecebe6"); ax.set_axisbelow(True)
    axes[0].set_xlabel("Share of rejected calls (%)"); axes[1].set_xlabel("Share of rejected calls (%)")
    fig.tight_layout(); fig.savefig(OUT / "fig2_failure_taxonomy.png", dpi=200); plt.close(fig)

# Fig 3: replay of first calls: recovered vs wrong place, per applier (pooled S runs)
def fig3():
    runs = [k for k in S if k.endswith("_target_S_s1") and not k.startswith("pilot")]
    agg = {}
    for k in runs:
        for ap, v in S[k]["replay_first_call"].items():
            a = agg.setdefault(ap, {"repaired_right_place": 0, "repaired_wrong_place": 0, "rejected": 0})
            for f in a:
                a[f] += v.get(f, 0)
    order = [("lineno", "+ strip line numbers"), ("ws", "+ whitespace / indent\n(unique only)"), ("fuzzy", "+ fuzzy >= 0.8\n(Aider-style)")]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    xs = range(len(order))
    good = [agg.get(a, {}).get("repaired_right_place", 0) for a, _ in order]
    bad = [agg.get(a, {}).get("repaired_wrong_place", 0) for a, _ in order]
    ax.bar([x - 0.18 for x in xs], good, width=0.34, color=BLUE, label="Recovered, right place", edgecolor="white", linewidth=2)
    ax.bar([x + 0.18 for x in xs], bad, width=0.34, color=ORANGE, label="Applied in the wrong place", edgecolor="white", linewidth=2)
    for x, g, b in zip(xs, good, bad):
        ax.text(x - 0.18, g + 0.5, str(g), ha="center", fontsize=8, color=INK)
        ax.text(x + 0.18, b + 0.5, str(b), ha="center", fontsize=8, color=INK)
    ax.set_xticks(list(xs)); ax.set_xticklabels([l for _, l in order], fontsize=8)
    ax.set_ylabel("First edit calls rejected by strict")
    ax.legend(frameon=False, fontsize=8); ax.yaxis.grid(True, color="#ecebe6"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT / "fig3_repair_tradeoff.png", dpi=200); plt.close(fig)

fig1(); fig2(); fig3()
print("figures written to", OUT)
