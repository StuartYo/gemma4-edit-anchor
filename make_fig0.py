"""Figure 0: what happens to one edit_file call under each condition (schematic)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#d9d8d2", "#0b0b0b", "#52514e"
fig, ax = plt.subplots(figsize=(9, 3.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
def box(x, y, w, h, text, fc="white", ec=GRAY, color=INK, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight="bold" if bold else "normal")
def arrow(x1, y1, x2, y2, text=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
    if text: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.2, text, ha="center", fontsize=7.5, color=INK2)
box(1, 16, 15, 8, "Model calls\nedit_file(old, new)", bold=True)
box(23, 16, 15, 8, "Exact, unique\nmatch?")
arrow(16.5, 20, 22.5, 20)
box(23, 30, 15, 6, "Apply edit", fc="#eaf2fc", ec=BLUE)
arrow(30.5, 24.5, 30.5, 29.5, "yes")
box(45, 16, 19, 8, "R / RN only:\nstrip line numbers, then\nline match ignoring indent\n(apply only if unique)", fs=7.5, fc="#e8f7f1", ec=AQUA)
arrow(38.5, 20, 44.5, 20, "no")
arrow(54.5, 24.5, 38.5, 33, "unique")
box(71, 26, 27, 8, "S / R: error text only\n\"old_string was not found\"", fs=8)
box(71, 8, 27, 12, "N / RN: error text plus the\nmost similar block of the file,\nwith line numbers\n(model copies it next turn)", fs=8, fc="#fdeee7", ec=ORANGE)
arrow(64.5, 20, 70.5, 29, "none or several")
arrow(64.5, 19, 70.5, 14)
ax.text(1, 3, "S and N skip the repair box: a failed exact match goes straight to the error. Every condition uses the same tool signature as the competition harness.", fontsize=7.5, color=INK2)
fig.tight_layout(); fig.savefig("figures/fig0_conditions.png", dpi=200)
print("ok")
