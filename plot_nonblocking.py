"""Read nonblocking_results.csv and produce plots/nonblocking_comparison.png.

Run from the project root after running run_nonblocking.sh:
    python3 plot_nonblocking.py

Expects nonblocking_results.csv with columns:
    config, params, mode, compute_ms, allreduce_ms, iter_ms, allreduce_pct

Where `mode` is one of {blocking, nonblocking} and `config` is one of
{50K, 235K, 567K, 1.46M}.
"""
import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

CSV_PATH = "nonblocking_results.csv"
OUT_PATH = "plots/nonblocking_comparison.png"

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

BLUE   = "#2563eb"
ORANGE = "#ea580c"


def main():
    if not os.path.exists(CSV_PATH):
        sys.exit(f"ERROR: {CSV_PATH} not found. Run run_nonblocking.sh first.")

    # Read CSV
    rows = []
    with open(CSV_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # Group by config preserving the four sizes in order
    config_order = ["50K", "235K", "567K", "1.46M"]
    blocking    = {}
    nonblocking = {}
    for row in rows:
        cfg = row["config"]
        try:
            iter_ms = float(row["iter_ms"])
        except ValueError:
            continue
        if row["mode"] == "blocking":
            blocking[cfg] = iter_ms
        elif row["mode"] == "nonblocking":
            nonblocking[cfg] = iter_ms

    # Validate
    for cfg in config_order:
        if cfg not in blocking or cfg not in nonblocking:
            print(f"WARN: missing data for {cfg}", file=sys.stderr)

    blocking_y    = [blocking.get(c, float("nan"))    for c in config_order]
    nonblocking_y = [nonblocking.get(c, float("nan")) for c in config_order]

    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(config_order))
    w = 0.35

    ax.bar(x - w/2, blocking_y,    w, color=ORANGE, edgecolor=ORANGE,
           linewidth=1.2, label="Blocking ring",       zorder=3)
    ax.bar(x + w/2, nonblocking_y, w, color=BLUE,   edgecolor=BLUE,
           linewidth=1.2, label="Non-blocking (pipelined)", zorder=3)

    for xi, b, nb in zip(x, blocking_y, nonblocking_y):
        if not np.isnan(b):
            ax.annotate(f"{b:.2f}", (xi - w/2, b),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=9, fontweight="bold",
                        color=ORANGE)
        if not np.isnan(nb):
            ax.annotate(f"{nb:.2f}", (xi + w/2, nb),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=9, fontweight="bold",
                        color=BLUE)
        if not (np.isnan(b) or np.isnan(nb)) and b > 0:
            speedup = (b - nb) / b * 100
            ax.annotate(f"-{speedup:.1f}%" if speedup > 0 else f"+{-speedup:.1f}%",
                        (xi, max(b, nb)),
                        textcoords="offset points", xytext=(0, 18),
                        ha="center", fontsize=10, fontweight="bold",
                        color="#16a34a" if speedup > 0 else "#9ca3af")

    ax.set_xticks(x)
    ax.set_xticklabels(config_order)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f} ms"))
    all_vals = [v for v in blocking_y + nonblocking_y if not np.isnan(v)]
    if all_vals:
        ax.set_ylim(0, max(all_vals) * 1.30)
    ax.set_xlabel("Network size (parameter count)")
    ax.set_ylabel("Avg time per iteration")
    ax.legend(loc="upper left", frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
