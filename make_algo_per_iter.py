"""Generate algo_per_iter.png for the report.

Per-iteration cost breakdown at P=8 on the 235K parameter network.
Compute (light blue, bottom) + all-reduce (red, top) stacked bars.

Run:
    python3 make_algo_per_iter.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

plt.rcParams.update({
    "font.family":          "DejaVu Sans",
    "font.size":            11,
    "axes.titlesize":       13,
    "axes.titleweight":     "bold",
    "axes.labelsize":       12,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.alpha":           0.3,
    "grid.linestyle":       "--",
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
})

BLUE       = "#2563eb"
BLUE_LIGHT = "#bfdbfe"
RED        = "#dc2626"

algos     = ["tree", "ring", "hd", "mpi\nbuiltin"]
compute   = [0.366, 0.336, 0.371, 0.337]
allreduce = [3.156, 1.697, 1.767, 1.522]
total     = [c + a for c, a in zip(compute, allreduce)]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
x = np.arange(len(algos))
w = 0.55

ax.bar(x, compute,   w, color=BLUE_LIGHT, edgecolor=BLUE, linewidth=1.2,
       label="Compute", zorder=3)
ax.bar(x, allreduce, w, bottom=compute,
       color=RED, edgecolor="#991b1b", linewidth=1.2,
       label="All-reduce", zorder=3)

for xi, t in zip(x, total):
    ax.annotate(f"{t:.2f} ms", (xi, t),
                textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=11, fontweight="bold",
                color="#111827")

ax.set_xticks(x)
ax.set_xticklabels(algos)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f} ms"))
ax.set_ylim(0, max(total) * 1.18)
ax.set_xlabel("All-reduce algorithm")
ax.set_ylabel("Avg time per iteration")
ax.legend(loc="upper right", frameon=False)

plt.tight_layout()
plt.savefig("plots/algo_per_iter.png", dpi=180, bbox_inches="tight",
            facecolor="white")
plt.close()
print("wrote plots/algo_per_iter.png")
