"""Cover image (1500 x 1000) for the Kaggle writeup, drawn from results/summary.json and
results/paired.json: Gemma 4 E2B, development set, strict (S) vs silent repair (R).

usage: python make_cover.py results/paired.json out.png
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = json.load(open(sys.argv[1], encoding="utf-8"))["E2B_dev100_R_vs_S"]
applied = (P["any_edit_applied"]["S"], P["any_edit_applied"]["R"])
solved = (P["solved"]["S"], P["solved"]["R"])
n = P["n_paired"]

BG, INK, INK2, BLUE, ORANGE, GRID = "#fbfaf7", "#151515", "#5c5b57", "#2a78d6", "#eb6834", "#e6e4dd"
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(7.5, 5.0), dpi=200, facecolor=BG)
fig.text(0.06, 0.90, "Fixing the Quote Did Not Fix the Edit", fontsize=21, weight="bold", color=INK)
fig.text(0.06, 0.835, "Exact-match edit_file failures of Gemma 4 E2B and E4B on a laptop GPU",
         fontsize=10.5, color=INK2)

ax = fig.add_axes([0.10, 0.19, 0.80, 0.54], facecolor=BG)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#c9c8c2")
xs = [0, 1]
for (a, b), color, label in [(applied, BLUE, "Tasks where an edit applied"), (solved, ORANGE, "Tasks solved exactly")]:
    ax.plot(xs, [a, b], color=color, lw=3.5, marker="o", ms=11, solid_capstyle="round")
    ax.text(-0.06, a, str(a), ha="right", va="center", fontsize=15, weight="bold", color=color)
    ax.text(1.06, b, str(b), ha="left", va="center", fontsize=15, weight="bold", color=color)
    ax.text(0.5, (a + b) / 2 + 4.5, label, ha="center", va="bottom", fontsize=10.5, color=color)
ax.set_xlim(-0.35, 1.35); ax.set_ylim(0, max(applied) + 14)
ax.set_xticks(xs); ax.set_xticklabels(["Strict matching\n(competition harness)", "Silent repair of\nwhitespace and line numbers"],
                                      fontsize=10, color=INK)
ax.set_yticks([]); ax.tick_params(axis="x", length=0, pad=8)
fig.text(0.06, 0.028, f"Gemma 4 E2B, {n} real single-file commits. McNemar p: applied {P['any_edit_applied']['mcnemar_p']:.2f}, "
         f"solved {P['solved']['mcnemar_p']:.2f}.\nNo repair condition, for either model, beat strict matching by more than seed noise.",
         fontsize=8.5, color=INK2)
fig.savefig(sys.argv[2], dpi=200, facecolor=BG)
print("cover written to", sys.argv[2])
