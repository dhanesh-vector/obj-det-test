"""
YOLOE HPO v5 – Visualization script.

Generates PNG plots for the hparams_v5 summary report.
Queries the Optuna SQLite database directly (no optuna import needed).

Usage (from repo root):
    python documentation/hparams_v5/generate_plots.py
"""

import json
import re
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "optuna_studies" / "yoloe_hparam_search_v5.db"
LOG_DIR = ROOT / "slurm-script" / "logs"
OUT_DIR = Path(__file__).resolve().parent

# ── Style (matches project's existing figures) ─────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "legend.fontsize":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
})

COLORS = {
    "complete": "#2166ac",   # blue
    "pruned":   "#d6604d",   # red-orange
    "best":     "#fdae61",   # amber
    "highlight":"#1a9641",   # green
}

# ── Top-5 fallback data (from YAML files) ─────────────────────────────────────
TOP5_RECORDS = [
    dict(rank=1, trial=409, slice_ap=0.7500, learning_rate=8.43e-4, weight_decay=3.96e-3,
         mosaic_prob=0.324, copy_paste_p=0.494, label_smooth=0.0786,
         batch_size=16, slice_stride=10,
         use_clahe=False, use_shift_scale_rotate=True,
         use_brightness_contrast=False, use_median_blur=True,
         median_blur_limit=3, use_speckle_noise=False),
    dict(rank=2, trial=123, slice_ap=0.7354, learning_rate=3.99e-4, weight_decay=8.18e-4,
         mosaic_prob=0.399, copy_paste_p=0.261, label_smooth=0.0866,
         batch_size=16, slice_stride=10,
         use_clahe=True,  use_shift_scale_rotate=True,
         use_brightness_contrast=False, use_median_blur=True,
         median_blur_limit=5, use_speckle_noise=False),
    dict(rank=3, trial=478, slice_ap=0.7289, learning_rate=1.06e-3, weight_decay=9.81e-4,
         mosaic_prob=0.485, copy_paste_p=0.666, label_smooth=0.0683,
         batch_size=16, slice_stride=10,
         use_clahe=False, use_shift_scale_rotate=True,
         use_brightness_contrast=False, use_median_blur=True,
         median_blur_limit=5, use_speckle_noise=True),
    dict(rank=4, trial=83,  slice_ap=0.7226, learning_rate=6.11e-4, weight_decay=1.31e-3,
         mosaic_prob=0.381, copy_paste_p=0.324, label_smooth=0.1032,
         batch_size=8,  slice_stride=10,
         use_clahe=True,  use_shift_scale_rotate=True,
         use_brightness_contrast=False, use_median_blur=True,
         median_blur_limit=7, use_speckle_noise=False),
    dict(rank=5, trial=305, slice_ap=0.7200, learning_rate=8.71e-4, weight_decay=5.76e-4,
         mosaic_prob=0.673, copy_paste_p=0.216, label_smooth=0.0889,
         batch_size=16, slice_stride=10,
         use_clahe=True,  use_shift_scale_rotate=True,
         use_brightness_contrast=False, use_median_blur=True,
         median_blur_limit=5, use_speckle_noise=False),
]
TOP5_DF = pd.DataFrame(TOP5_RECORDS)


# ══════════════════════════════════════════════════════════════════════════════
# Database loader
# ══════════════════════════════════════════════════════════════════════════════

def _decode_categorical(dist_json: str, raw_value: float):
    """Decode an Optuna categorical param value (index → actual choice)."""
    try:
        d = json.loads(dist_json)
        # Optuna 3.x / 4.x formats
        choices = None
        if isinstance(d, dict):
            if "name" in d and d["name"] == "CategoricalDistribution":
                choices = d.get("attributes", {}).get("choices")
            elif "CategoricalDistribution" in d:
                choices = d["CategoricalDistribution"].get("choices")
        if choices is not None:
            return choices[int(raw_value)]
    except Exception:
        pass
    return raw_value


def load_study_data() -> pd.DataFrame:
    """
    Return a DataFrame with one row per trial, columns:
      trial_number, state_name, value, <param_name>...
    """
    conn = sqlite3.connect(str(DB_PATH))

    trials = pd.read_sql(
        "SELECT trial_id, number AS trial_number, state FROM trials WHERE study_id = 1",
        conn,
    )
    values = pd.read_sql("SELECT trial_id, value FROM trial_values", conn)
    params_raw = pd.read_sql(
        "SELECT trial_id, param_name, param_value, distribution_json FROM trial_params",
        conn,
    )
    conn.close()

    trials = trials.merge(values, on="trial_id", how="left")

    records = []
    for _, row in params_raw.iterrows():
        records.append({
            "trial_id":  row["trial_id"],
            "param_name": row["param_name"],
            "decoded":   _decode_categorical(row["distribution_json"], row["param_value"]),
        })
    if records:
        params_long = pd.DataFrame(records)
        params_pivot = (
            params_long
            .pivot_table(index="trial_id", columns="param_name",
                         values="decoded", aggfunc="first")
            .reset_index()
        )
        trials = trials.merge(params_pivot, on="trial_id", how="left")

    # Optuna stores state as uppercase strings ("COMPLETE", "PRUNED", etc.)
    trials["state_name"] = trials["state"].str.lower().fillna("unknown")
    return trials


# ══════════════════════════════════════════════════════════════════════════════
# Log parser
# ══════════════════════════════════════════════════════════════════════════════

_RE_TRIAL  = re.compile(r"\[Trial\s+(\d+)\].*?lr=([\d.e+\-]+).*?wd=([\d.e+\-]+)")
_RE_EPOCH  = re.compile(
    r"Epoch\s+(\d+)/\d+\s*\|.*?TLoss=([\d.]+).*?VLoss=([\d.]+)"
    r".*?AP@50=([\d.]+).*?SliceAP=([\d.]+).*?EMA_SliceAP=([\d.]+)"
)


def load_epoch_data() -> dict:
    """Parse SLURM log files → {trial_number: {"lr", "wd", "df"}} ."""
    all_trials: dict = {}
    for lf in sorted(LOG_DIR.glob("yoloe_tune_v5_916_*.out")):
        current = None
        with open(lf) as f:
            for line in f:
                m = _RE_TRIAL.search(line)
                if m:
                    current = int(m.group(1))
                    if current not in all_trials:
                        all_trials[current] = {
                            "lr": float(m.group(2)),
                            "wd": float(m.group(3)),
                            "rows": [],
                        }
                    continue
                m = _RE_EPOCH.search(line)
                if m and current is not None and current in all_trials:
                    all_trials[current]["rows"].append({
                        "epoch":        int(m.group(1)),
                        "train_loss":   float(m.group(2)),
                        "val_loss":     float(m.group(3)),
                        "ap50":         float(m.group(4)),
                        "slice_ap":     float(m.group(5)),
                        "ema_slice_ap": float(m.group(6)),
                    })
    result = {}
    for t, v in all_trials.items():
        if v["rows"]:
            result[t] = {"lr": v["lr"], "wd": v["wd"], "df": pd.DataFrame(v["rows"])}
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ══════════════════════════════════════════════════════════════════════════════

def _savefig(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ── Plot 1: Optimisation history ───────────────────────────────────────────────
def plot_optim_history(df: pd.DataFrame):
    completed = df[df["state_name"] == "complete"].copy()
    pruned    = df[df["state_name"] == "pruned"].copy()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.scatter(pruned["trial_number"],    pruned["value"],
               s=12, alpha=0.30, color=COLORS["pruned"],
               label=f"Pruned  ({len(pruned)})", zorder=1, rasterized=True)
    ax.scatter(completed["trial_number"], completed["value"],
               s=18, alpha=0.65, color=COLORS["complete"],
               label=f"Completed ({len(completed)})", zorder=2, rasterized=True)

    if not completed.empty:
        best_so_far = (
            completed.sort_values("trial_number")
            .assign(best=lambda d: d["value"].cummax())
        )
        ax.plot(best_so_far["trial_number"], best_so_far["best"],
                color=COLORS["best"], lw=2.0, label="Best so far", zorder=3)
        ax.axhline(completed["value"].max(),
                   color=COLORS["highlight"], ls="--", lw=1.5,
                   label=f"Best = {completed['value'].max():.4f}", zorder=4)

    ax.set_xlabel("Trial Number")
    ax.set_ylabel("EMA SliceAP@50")
    ax.set_title("HPO v5 – Optimisation History  (300 trials · 20 workers × 15)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _savefig(fig, "plot_01_optim_history.png")


# ── Plot 2: Top-5 hyperparameter comparison ─────────────────────────────────
def plot_top5_comparison():
    params = [
        ("learning_rate", "Learning Rate"),
        ("weight_decay",  "Weight Decay"),
        ("mosaic_prob",   "Mosaic Prob"),
        ("copy_paste_p",  "Copy-Paste P"),
        ("label_smooth",  "Label Smooth"),
    ]
    labels_y = [f"#{r} T{t}" for r, t in zip(TOP5_DF["rank"], TOP5_DF["trial"])]
    palette   = sns.color_palette("Set2", len(TOP5_DF))

    fig, axes = plt.subplots(1, len(params), figsize=(16, 4))
    for ax, (col, title) in zip(axes, params):
        vals = TOP5_DF[col].values
        bars = ax.barh(labels_y, vals, color=palette)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        # value labels
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_width() * 1.03, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", ha="left", fontsize=8,
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("HPO v5 – Top-5 Trial Hyperparameter Comparison",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "plot_02_top5_comparison.png")


# ── Plot 3: Epoch learning curves (sample trials from logs) ───────────────────
def plot_epoch_curves(epoch_data: dict):
    if not epoch_data:
        print("  Skipping plot_03: no epoch data parsed from logs.")
        return

    # Pick: best trial (409) + 2 highest-EMA trials we have in logs
    target = 409
    ranked = sorted(
        epoch_data.keys(),
        key=lambda t: epoch_data[t]["df"]["ema_slice_ap"].max(),
        reverse=True,
    )
    pool   = [target] + [t for t in ranked if t != target]
    chosen = []
    for t in pool:
        if t in epoch_data and len(epoch_data[t]["df"]) >= 5:
            chosen.append(t)
        if len(chosen) == 3:
            break
    if not chosen:
        chosen = list(epoch_data.keys())[:3]

    n = len(chosen)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), sharex="col")
    if n == 1:
        axes = axes.reshape(2, 1)

    row_colors = sns.color_palette("deep", n)
    for i, trial_num in enumerate(chosen):
        td  = epoch_data[trial_num]
        edf = td["df"]
        c   = row_colors[i]
        title = f"Trial {trial_num}\nlr={td['lr']:.2e}  wd={td['wd']:.2e}"

        # Row 0 – losses
        axes[0, i].plot(edf["epoch"], edf["train_loss"],
                        color=c, ls="-",  lw=1.8, label="Train Loss")
        axes[0, i].plot(edf["epoch"], edf["val_loss"],
                        color=c, ls="--", lw=1.8, alpha=0.7, label="Val Loss")
        axes[0, i].set_yscale("log")
        axes[0, i].set_title(title, fontsize=9)
        axes[0, i].set_ylabel("Loss (log)")
        axes[0, i].legend(fontsize=8)

        # Row 1 – metrics
        axes[1, i].plot(edf["epoch"], edf["slice_ap"],
                        color="steelblue",  ls="-",  lw=1.5, label="SliceAP")
        axes[1, i].plot(edf["epoch"], edf["ema_slice_ap"],
                        color="darkorange", ls="-",  lw=2.2, label="EMA SliceAP")
        axes[1, i].plot(edf["epoch"], edf["ap50"],
                        color="seagreen",   ls=":",  lw=1.5, alpha=0.8, label="AP@50")
        axes[1, i].set_ylim(0, 1)
        axes[1, i].set_xlabel("Epoch")
        axes[1, i].set_ylabel("Score")
        axes[1, i].legend(fontsize=8)

    fig.suptitle("HPO v5 – Training Dynamics  (sample trials)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "plot_03_epoch_curves.png")


# ── Plot 4: Continuous hyperparameter vs SliceAP ──────────────────────────────
def plot_continuous_params(df: pd.DataFrame):
    completed = df[df["state_name"] == "complete"].dropna(subset=["value"]).copy()
    if completed.empty:
        print("  Skipping plot_04: no completed trial data.")
        return

    cont_cols = [
        ("learning_rate", "Learning Rate",  True),
        ("weight_decay",  "Weight Decay",   True),
        ("mosaic_prob",   "Mosaic Prob",     False),
        ("copy_paste_p",  "Copy-Paste P",   False),
        ("label_smooth",  "Label Smooth",   False),
    ]
    available = [(c, l, lg) for c, l, lg in cont_cols if c in completed.columns]
    n = len(available)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()

    threshold = completed["value"].quantile(0.80)
    completed["top_20"] = completed["value"] >= threshold

    for idx, (col, label, log_scale) in enumerate(available):
        ax = axes_flat[idx]
        x  = pd.to_numeric(completed[col], errors="coerce")
        if log_scale:
            x = np.log10(x)
            xlabel = f"log₁₀({label})"
        else:
            xlabel = label

        low  = ~completed["top_20"]
        high =  completed["top_20"]

        ax.scatter(x[low],  completed.loc[low,  "value"],
                   s=12, alpha=0.28, color=COLORS["complete"],
                   label="Bottom 80%", rasterized=True)
        ax.scatter(x[high], completed.loc[high, "value"],
                   s=20, alpha=0.75, color=COLORS["best"],
                   label="Top 20%",   rasterized=True)

        # mark top-5 trials
        if col in completed.columns:
            top5_rows = completed.nlargest(5, "value")
            x5 = pd.to_numeric(top5_rows[col], errors="coerce")
            if log_scale:
                x5 = np.log10(x5)
            ax.scatter(x5, top5_rows["value"],
                       s=80, color=COLORS["highlight"], marker="*",
                       label="Top 5", zorder=5)

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("EMA SliceAP@50", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        if idx == 0:
            ax.legend(fontsize=8)

    # hide the 6th unused panel
    axes_flat[-1].set_visible(False)

    fig.suptitle("HPO v5 – Continuous Hyperparameters vs EMA SliceAP@50\n"
                 "(all completed trials · stars = top-5)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "plot_04_continuous_params.png")


# ── Plot 5: Categorical hyperparameter box plots ───────────────────────────────
def plot_categorical_analysis(df: pd.DataFrame):
    completed = df[df["state_name"] == "complete"].dropna(subset=["value"]).copy()
    if completed.empty:
        print("  Skipping plot_05: no completed trial data.")
        return

    cat_cols = [
        ("batch_size",          "Batch Size"),
        ("slice_stride",        "Slice Stride"),
        ("use_clahe",           "Use CLAHE"),
        ("use_brightness_contrast", "Brightness Contrast"),
        ("use_speckle_noise",   "Speckle Noise"),
        ("median_blur_limit",   "Median Blur Limit"),
    ]
    available = [(c, l) for c, l in cat_cols if c in completed.columns]
    n = len(available)
    if n == 0:
        print("  Skipping plot_05: no categorical columns found.")
        return

    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten()

    for idx, (col, label) in enumerate(available):
        ax = axes_flat[idx]
        plot_df = completed[[col, "value"]].copy()
        plot_df[col] = plot_df[col].astype(str)

        def _sort_key(x):
            try:
                return (0, float(x))
            except ValueError:
                return (1, x)
        order = sorted(plot_df[col].dropna().unique(), key=_sort_key)

        sns.boxplot(
            data=plot_df, x=col, y="value", order=order,
            hue=col, palette="Set2", legend=False,
            width=0.55, ax=ax, linewidth=1.2,
        )
        ax.set_xlabel("")
        ax.set_ylabel("EMA SliceAP@50" if idx % ncols == 0 else "")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)

        for j, cat in enumerate(order):
            n_cat = (plot_df[col] == cat).sum()
            y_min = ax.get_ylim()[0]
            ax.text(j, y_min + 0.002, f"n={n_cat}",
                    ha="center", fontsize=7.5, color="dimgray")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("HPO v5 – Categorical Hyperparameter Performance\n"
                 "(EMA SliceAP@50 by category, all completed trials)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "plot_05_categorical_analysis.png")


# ── Plot 6: LR vs Weight-Decay 2D scatter ─────────────────────────────────────
def plot_lr_wd_scatter(df: pd.DataFrame):
    completed = df[df["state_name"] == "complete"].dropna(subset=["value"]).copy()
    if "learning_rate" not in completed.columns or "weight_decay" not in completed.columns:
        print("  Skipping plot_06: lr/wd columns not found.")
        return

    completed["log_lr"] = np.log10(pd.to_numeric(completed["learning_rate"], errors="coerce"))
    completed["log_wd"] = np.log10(pd.to_numeric(completed["weight_decay"],  errors="coerce"))
    completed = completed.dropna(subset=["log_lr", "log_wd"])

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        completed["log_lr"], completed["log_wd"],
        c=completed["value"], cmap="RdYlGn",
        s=20, alpha=0.65, vmin=0.3, vmax=0.75, rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax, label="EMA SliceAP@50")

    # mark top-5
    top5 = completed.nlargest(5, "value")
    ax.scatter(top5["log_lr"], top5["log_wd"],
               s=120, c="none", edgecolors="black", linewidths=1.5,
               zorder=5, label="Top 5")
    for _, row in top5.iterrows():
        ax.annotate(
            f"T{int(row['trial_number'])} {row['value']:.4f}",
            xy=(row["log_lr"], row["log_wd"]),
            xytext=(4, 4), textcoords="offset points", fontsize=7,
        )

    # search-space boundary boxes
    ax.axvline(np.log10(1e-4), color="gray", ls=":", lw=1, alpha=0.5)
    ax.axvline(np.log10(5e-3), color="gray", ls=":", lw=1, alpha=0.5)
    ax.axhline(np.log10(1e-4), color="gray", ls=":", lw=1, alpha=0.5)
    ax.axhline(np.log10(5e-2), color="gray", ls=":", lw=1, alpha=0.5)

    ax.set_xlabel("log₁₀(Learning Rate)")
    ax.set_ylabel("log₁₀(Weight Decay)")
    ax.set_title("HPO v5 – Learning Rate vs Weight Decay\n"
                 "(colour = EMA SliceAP@50, dashed = search bounds, ○ = top 5)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, "plot_06_lr_wd_scatter.png")


# ── Plot 7: Augmentation flag heatmap (top-30 trials) ──────────────────────────
def plot_augmentation_heatmap(df: pd.DataFrame, top_n: int = 30):
    completed = df[df["state_name"] == "complete"].dropna(subset=["value"])
    top = completed.nlargest(top_n, "value").copy()
    if top.empty:
        print("  Skipping plot_07: no completed trial data.")
        return

    flag_map = {
        "use_clahe":                "CLAHE",
        "use_shift_scale_rotate":   "ShiftScaleRotate",
        "use_brightness_contrast":  "BrightnessContrast",
        "use_median_blur":          "MedianBlur",
        "use_speckle_noise":        "SpeckleNoise",
    }
    available = {k: v for k, v in flag_map.items() if k in top.columns}
    if not available:
        print("  Skipping plot_07: augmentation flag columns not found in DB.")
        return

    heat = top[list(available.keys())].copy()
    heat.columns = list(available.values())
    for col in heat.columns:
        heat[col] = pd.to_numeric(
            heat[col].map({True: 1, False: 0, 1.0: 1, 0.0: 0,
                           "True": 1, "False": 0}),
            errors="coerce",
        )
    heat.index = [
        f"T{int(n)}  {v:.4f}"
        for n, v in zip(top["trial_number"], top["value"])
    ]

    fig_h = max(7, top_n * 0.38)
    fig, ax = plt.subplots(figsize=(7, fig_h))
    sns.heatmap(
        heat, annot=True, fmt=".0f", cmap="RdYlGn",
        vmin=0, vmax=1, linewidths=0.4, linecolor="lightgray",
        ax=ax, cbar_kws={"label": "Enabled  (1 = Yes)"},
    )
    ax.set_xlabel("Augmentation flag")
    ax.set_ylabel(f"Top-{top_n} trials  (trial # · SliceAP)")
    ax.set_title(f"HPO v5 – Augmentation Choices for Top-{top_n} Trials",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "plot_07_augmentation_heatmap.png")


# ── Plot 8: Version comparison bar ────────────────────────────────────────────
def plot_version_comparison():
    """Bar chart comparing best SliceAP@50 across HPO v3, v4, v5."""
    versions = ["v3 (SLURM 732)", "v4 (SLURM 752)", "v5 (SLURM 916)"]
    best_aps  = [0.6187,          0.7280,            0.7500]
    n_trials  = [300,             300,               300]
    colors    = ["#9ecae1", "#4292c6", "#084594"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(versions, best_aps, color=colors, width=0.5, alpha=0.9)
    for bar, val in zip(bars, best_aps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    # delta annotations
    for i in range(1, len(best_aps)):
        delta = best_aps[i] - best_aps[i - 1]
        mid_x = (bars[i - 1].get_x() + bars[i - 1].get_width() +
                 bars[i].get_x()) / 2
        y     = max(best_aps[i - 1], best_aps[i]) + 0.02
        ax.annotate(
            f"Δ +{delta:.4f}",
            xy=(mid_x, y), ha="center", fontsize=9, color="dimgray",
        )

    ax.set_ylabel("Best EMA SliceAP@50")
    ax.set_ylim(0, 0.85)
    ax.set_title("HPO Study Progression: v3 → v4 → v5\n"
                 "(300 Optuna trials each, TPE + Hyperband pruning)",
                 fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _savefig(fig, "plot_08_version_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Loading Optuna study from: {DB_PATH}")
    try:
        df = load_study_data()
        n_complete = (df["state_name"] == "complete").sum()
        n_pruned   = (df["state_name"] == "pruned").sum()
        print(f"  {len(df)} trials total: {n_complete} completed, {n_pruned} pruned")
        db_ok = True
    except Exception as exc:
        print(f"  WARNING: DB load failed ({exc}). DB-dependent plots will be skipped.")
        df    = pd.DataFrame()
        db_ok = False

    print(f"\nParsing epoch data from {LOG_DIR} ...")
    epoch_data = load_epoch_data()
    print(f"  Found epoch data for {len(epoch_data)} trials")

    print(f"\nGenerating plots → {OUT_DIR}/")

    # Always-available plots
    plot_top5_comparison()
    plot_version_comparison()

    # Epoch curves from log files
    plot_epoch_curves(epoch_data)

    # DB-dependent plots
    if db_ok and not df.empty:
        plot_optim_history(df)
        plot_continuous_params(df)
        plot_categorical_analysis(df)
        plot_lr_wd_scatter(df)
        plot_augmentation_heatmap(df)
    else:
        print("  Skipping DB-based plots (plots 01, 04, 05, 06, 07).")

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()
