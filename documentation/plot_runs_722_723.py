"""
Generate comparison plots for baseline (no PU loss) vs PU loss training runs.
Saves PDFs to documentation/figures/.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Load data ─────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

RUN_FILES = {
    "Baseline (No PU Loss)": "training_run_20260316_105425.json",
    "PU Loss":               "training_run_20260316_175342.json",
}

COLORS = {
    "Baseline (No PU Loss)": "#2166ac",
    "PU Loss":               "#d6604d",
}
LINESTYLES = {
    "Baseline (No PU Loss)": "-",
    "PU Loss":               "--",
}

data = {}
for label, fname in RUN_FILES.items():
    with open(RESULTS_DIR / fname) as f:
        data[label] = json.load(f)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

def make_epochs(metrics_list):
    return list(range(1, len(metrics_list) + 1))


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Training / Validation loss + AP@50 + SliceAP + Precision + Recall
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.suptitle(
    "Training Curves: Baseline vs PU Loss",
    fontsize=13, fontweight="bold", y=1.01,
)

panels = [
    ("train_loss",  "Training Loss",    "Loss",       True),
    ("val_loss",    "Validation Loss",  "Loss",       True),
    ("map_50",      "Val AP@50",        "AP@50",      False),
    ("precision",   "Val Precision",    "Precision",  False),
    ("recall",      "Val Recall",       "Recall",     False),
    ("slice_ap_50", "Val Slice-AP@50",  "Slice AP@50",False),
]

for ax, (key, title, ylabel, logy) in zip(axes.flat, panels):
    for label, d in data.items():
        m = d["metrics"][key]
        ep = make_epochs(m)
        ax.plot(ep, m,
                color=COLORS[label], linestyle=LINESTYLES[label],
                linewidth=1.8, label=label, alpha=0.9)
        if key == "map_50":
            best_ep  = d["best_epoch"]
            best_val = d["best_ap_50"]
            ax.axvline(best_ep, color=COLORS[label], linestyle=":",
                       linewidth=1.2, alpha=0.6)
            ax.scatter([best_ep], [best_val],
                       color=COLORS[label], zorder=5, s=60, marker="*")
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")

fig.tight_layout()
p1 = FIGURES_DIR / "pu_loss_training_curves.pdf"
fig.savefig(p1, bbox_inches="tight")
print(f"Saved: {p1}")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: mAP family curves (AP@50, AP@75, mAP, MAR@100)
# ═══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
fig2.suptitle("Detection Metrics: Baseline vs PU Loss", fontsize=13, fontweight="bold")

map_panels = [
    ("map_50",  "AP@50"),
    ("map_75",  "AP@75"),
    ("map",     "mAP (0.5:0.95)"),
    ("mar_100", "MAR@100"),
]

for ax, (key, title) in zip(axes2, map_panels):
    for label, d in data.items():
        m = d["metrics"][key]
        ax.plot(make_epochs(m), m,
                color=COLORS[label], linestyle=LINESTYLES[label],
                linewidth=1.8, label=label, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.legend(loc="best", fontsize=8)

fig2.tight_layout()
p2 = FIGURES_DIR / "pu_loss_map_metrics.pdf"
fig2.savefig(p2, bbox_inches="tight")
print(f"Saved: {p2}")
plt.close(fig2)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Summary bar chart — peak values
# ═══════════════════════════════════════════════════════════════════════════════
summary_keys = [
    ("map_50",      "Best AP@50"),
    ("map_75",      "Best AP@75"),
    ("map",         "Best mAP\n(0.5:0.95)"),
    ("slice_ap_50", "Best\nSlice-AP@50"),
    ("precision",   "Best Precision"),
    ("recall",      "Best Recall"),
    ("mar_100",     "Best MAR@100"),
]

x = np.arange(len(summary_keys))
width = 0.35

fig3, ax3 = plt.subplots(figsize=(13, 5))
for i, (run_label, d) in enumerate(data.items()):
    vals = [float(np.max(d["metrics"][k])) for k, _ in summary_keys]
    offset = (i - 0.5) * width
    bars = ax3.bar(x + offset, vals, width,
                   label=run_label, color=COLORS[run_label], alpha=0.85)
    for bar, v in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.006,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)

ax3.set_xticks(x)
ax3.set_xticklabels([lbl for _, lbl in summary_keys], fontsize=9)
ax3.set_ylabel("Score")
ax3.set_ylim(0, 1.05)
ax3.set_title("Peak Metric Comparison: Baseline vs PU Loss",
              fontsize=12, fontweight="bold")
ax3.legend()
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.yaxis.grid(True, linestyle="--", alpha=0.4)
ax3.set_axisbelow(True)

fig3.tight_layout()
p3 = FIGURES_DIR / "pu_loss_summary_bar.pdf"
fig3.savefig(p3, bbox_inches="tight")
print(f"Saved: {p3}")
plt.close(fig3)

print("All figures generated successfully.")
