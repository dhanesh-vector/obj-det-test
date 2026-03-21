"""
Evaluate the best model on the held-out test set.

Outputs (saved to inference/test_results/):
  metrics.json                  — all metrics (machine-readable)
  metrics_report.txt            — human-readable metrics table
  plot_01_metrics_summary.png   — bar chart of all key metrics
  plot_02_pr_curve.png          — precision-recall curve with operating points
  plot_03_score_dist.png        — TP vs FP confidence score distribution
  plot_04_lesion_breakdown.png  — detected / missed / false-alarm summary
  plot_05_predictions_grid.png  — sample test images with annotated predictions
  plot_06_per_scan_breakdown.png — per-scan TP/FN/FP bar chart
  plot_07_scan_slice_heatmap.png — per-slice detection status across each scan

Usage (from repo root):
    python inference/test_eval.py \\
        --weights checkpoints/best_model_20260318_083558.pth \\
        --data-dir /projects/tenomix/ml-share/training/07/data \\
        --split test
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
import yaml
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from model.yoloe import build_yoloe, YOLOEWithLoss
from data.dataset import UltrasoundDataset, collate_fn
from utils.metrics import decode_predictions, Evaluator

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
})

# ── Constants ──────────────────────────────────────────────────────────────────
CONF_THRESH_EVAL  = 0.05   # low threshold → pass all candidates to Evaluator
# CONF_THRESH_FIXED and IOU_THRESH are set from CLI args in main()
CONF_THRESH_FIXED = 0.50   # default; overridden by --conf-thresh
IOU_THRESH        = 0.50   # default; overridden by --iou-thresh


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(weights_path: str, device: torch.device,
               model_size: str = "s",
               num_classes: int = 1,
               use_pu_loss: bool = False) -> YOLOEWithLoss:
    base_model = build_yoloe(model_size=model_size, num_classes=num_classes)
    model = YOLOEWithLoss(model=base_model, num_classes=num_classes,
                          use_pu_loss=use_pu_loss)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    try:
        model.model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(state)
    # eval() disables dropout and switches BatchNorm to use running statistics
    # rather than batch statistics — required for correct single-image inference.
    model.to(device).eval()
    print(f"Loaded checkpoint : {weights_path}")
    print(f"  model_size      : {model_size}")
    print(f"  use_pu_loss     : {use_pu_loss}")
    print(f"  Trained to epoch: {ckpt.get('epoch', '?')}")
    print(f"  Saved val AP@50 : {ckpt.get('ap_50', float('nan')):.4f}")
    print(f"  Saved SliceAP   : {ckpt.get('slice_ap_50', float('nan')):.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Inference loop
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model, dataloader, device):
    """
    Run model on the full dataloader.

    Returns
    -------
    all_preds   : list of {boxes, scores, labels} per image
    all_targets : list of {boxes, labels, image_id} per image
    evaluator   : populated Evaluator (call .compute() for metrics)
    """
    evaluator = Evaluator()
    all_preds, all_targets = [], []

    # eval() called again here as a safeguard — ensures BN uses running stats
    # even if the caller forgot to set it.  No augmentations are applied:
    # UltrasoundDataset is constructed with transform=None, mosaic_prob=0.0,
    # copy_paste=None (all defaults), so images are loaded and resized only.
    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            device_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            cls_scores, reg_dists = model(images)

            h, w = images.shape[2:]
            feat_sizes = [(h // s, w // s) for s in model.model.strides]
            anchor_points, stride_tensor = model.model.get_anchor_points(
                feat_sizes, device, images.dtype
            )
            batch_preds = decode_predictions(
                cls_scores, reg_dists, anchor_points, stride_tensor,
                conf_thresh=CONF_THRESH_EVAL, iou_thresh=0.5,
            )

            evaluator.update(batch_preds, device_targets)
            all_preds.extend(batch_preds)
            all_targets.extend(device_targets)

    return all_preds, all_targets, evaluator


# ══════════════════════════════════════════════════════════════════════════════
# Per-image lesion breakdown
# ══════════════════════════════════════════════════════════════════════════════

def compute_lesion_breakdown(all_preds, all_targets,
                             conf_thresh=CONF_THRESH_FIXED,
                             iou_thresh=IOU_THRESH):
    """
    For every image compute lesion-level TP / FN / FP counts.

    A GT box is:
      - Detected (TP)  → matched by a prediction with IoU ≥ iou_thresh and
                          score ≥ conf_thresh
      - Missed   (FN)  → no matching prediction

    A prediction box is:
      - True Positive  → matched a GT box (as above)
      - False Alarm (FP) → no matching GT box

    Returns
    -------
    per_image : list of dicts, one per image:
        {n_gt, tp, fn, fp,
         pred_boxes_tp, pred_boxes_fp,   # tensors for visualisation
         gt_boxes_tp, gt_boxes_fn}
    totals    : dict with aggregate counts
    """
    per_image = []
    total_gt = total_tp = total_fn = total_fp = 0

    for pred, target in zip(all_preds, all_targets):
        gt_boxes    = target["boxes"].cpu()
        pred_boxes  = pred["boxes"].cpu()
        pred_scores = pred["scores"].cpu()

        # Apply fixed confidence threshold
        keep = pred_scores >= conf_thresh
        pred_boxes  = pred_boxes[keep]
        pred_scores = pred_scores[keep]

        n_gt   = len(gt_boxes)
        n_pred = len(pred_boxes)

        tp_pred_mask = torch.zeros(n_pred, dtype=torch.bool)
        tp_gt_mask   = torch.zeros(n_gt,   dtype=torch.bool)

        if n_gt > 0 and n_pred > 0:
            iou = box_iou(pred_boxes, gt_boxes)   # (n_pred, n_gt)
            matched_preds = set()
            matched_gts   = set()
            # Greedy match: highest IoU pairs first
            for flat_idx in iou.flatten().argsort(descending=True):
                pi = (flat_idx // n_gt).item()
                gi = (flat_idx  % n_gt).item()
                if iou[pi, gi] < iou_thresh:
                    break
                if pi not in matched_preds and gi not in matched_gts:
                    matched_preds.add(pi)
                    matched_gts.add(gi)
                    tp_pred_mask[pi] = True
                    tp_gt_mask[gi]   = True

        tp = int(tp_pred_mask.sum())
        fp = n_pred - tp
        fn = n_gt - int(tp_gt_mask.sum())

        per_image.append({
            "n_gt":          n_gt,
            "tp":            tp,
            "fn":            fn,
            "fp":            fp,
            "pred_boxes_tp": pred_boxes[tp_pred_mask],
            "pred_boxes_fp": pred_boxes[~tp_pred_mask],
            "gt_boxes_tp":   gt_boxes[tp_gt_mask]   if n_gt > 0 else gt_boxes,
            "gt_boxes_fn":   gt_boxes[~tp_gt_mask]  if n_gt > 0 else gt_boxes,
            "pred_scores":   pred_scores,
        })

        total_gt += n_gt
        total_tp += tp
        total_fn += fn
        total_fp += fp

    totals = {
        "total_gt_lesions":    total_gt,
        "detected_tp":         total_tp,
        "missed_fn":           total_fn,
        "false_alarms_fp":     total_fp,
        "detection_rate":      total_tp / total_gt if total_gt > 0 else 0.0,
        "miss_rate":           total_fn / total_gt if total_gt > 0 else 0.0,
        "false_alarm_per_img": total_fp / len(all_preds) if all_preds else 0.0,
        "conf_thresh":         conf_thresh,
        "iou_thresh":          iou_thresh,
    }
    return per_image, totals


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def _savefig(fig, path: Path, name: str):
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


def plot_metrics_summary(metrics: dict, out_dir: Path,
                         conf_thresh: float = 0.50, iou_thresh: float = 0.50):
    keys = [
        ("map_50",                  "AP@50"),
        ("map_75",                  "AP@75"),
        ("map",                     "mAP@50:95"),
        ("map_35",                  "AP@35"),
        ("slice_ap_50",             "SliceAP@50"),
        ("slice_ap_75",             "SliceAP@75"),
        ("precision",               f"Precision\n(conf≥{conf_thresh})"),
        ("recall",                  f"Recall\n(conf≥{conf_thresh})"),
        ("f1",                      f"F1\n(conf≥{conf_thresh})"),
        ("precision_at_recall80",   "P @ R=0.8"),
        ("recall_at_precision80",   "R @ P=0.8"),
    ]
    labels = [l for k, l in keys if k in metrics]
    values = [metrics[k] for k, _ in keys if k in metrics]

    colors = ["#2166ac" if v >= 0.6 else "#4393c3" if v >= 0.4 else "#92c5de"
              for v in values]

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(labels, values, color=colors, width=0.6, alpha=0.9)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Test Set Metrics — Best Model (epoch 28)",
                 fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _savefig(fig, out_dir, "plot_01_metrics_summary.png")


def plot_pr_curve(evaluator: Evaluator, metrics: dict, out_dir: Path,
                  conf_thresh: float = 0.50, iou_thresh: float = 0.50):
    prec, rec, scores = evaluator.pr_curve_data(iou_thresh=iou_thresh)

    order  = np.argsort(rec)
    rec_s  = rec[order]
    prec_s = prec[order]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.fill_between(rec_s, prec_s, alpha=0.12, color="steelblue")
    ax.plot(rec_s, prec_s, color="steelblue", lw=2,
            label=f"AP@50 = {metrics.get('map_50', 0):.3f}")

    # Mark P@R=0.8
    p_at_r80 = metrics.get("precision_at_recall80", None)
    if p_at_r80 is not None:
        ax.axvline(0.8,  color="tomato",     ls="--", lw=1.2, alpha=0.8)
        ax.scatter([0.8], [p_at_r80], s=80, color="tomato", zorder=5,
                   label=f"P @ R=0.8 = {p_at_r80:.3f}")

    # Mark R@P=0.8
    r_at_p80 = metrics.get("recall_at_precision80", None)
    if r_at_p80 is not None:
        ax.axhline(0.8, color="darkorange", ls="--", lw=1.2, alpha=0.8)
        ax.scatter([r_at_p80], [0.8], s=80, color="darkorange", zorder=5,
                   label=f"R @ P=0.8 = {r_at_p80:.3f}")

    # Mark operating point at fixed conf threshold
    p05 = metrics.get("precision", None)
    r05 = metrics.get("recall",    None)
    if p05 is not None and r05 is not None:
        ax.scatter([r05], [p05], s=100, color="seagreen", marker="D", zorder=6,
                   label=f"conf≥{conf_thresh}  P={p05:.3f}  R={r05:.3f}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (IoU = {iou_thresh}) — Test Set",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    _savefig(fig, out_dir, "plot_02_pr_curve.png")


def plot_score_distribution(evaluator: Evaluator, out_dir: Path,
                            conf_thresh: float = 0.50, iou_thresh: float = 0.50):
    tp_scores, fp_scores = evaluator.score_distributions(iou_thresh=iou_thresh)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 31)

    if tp_scores:
        ax.hist(tp_scores, bins=bins, color="steelblue", alpha=0.7,
                label=f"True Positives  (n={len(tp_scores)})", density=True)
    if fp_scores:
        ax.hist(fp_scores, bins=bins, color="tomato", alpha=0.7,
                label=f"False Positives (n={len(fp_scores)})", density=True)

    ax.axvline(conf_thresh, color="black", ls="--", lw=1.5,
               label=f"Threshold = {conf_thresh}")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Density")
    ax.set_title("TP vs FP Confidence Score Distribution — Test Set",
                 fontsize=11, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, out_dir, "plot_03_score_dist.png")


def plot_lesion_breakdown(totals: dict, per_image: list, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel A: Overall pie ──────────────────────────────────────────────────
    ax = axes[0]
    tp = totals["detected_tp"]
    fn = totals["missed_fn"]
    fp = totals["false_alarms_fp"]
    sizes  = [tp, fn, fp]
    labels = [f"Detected (TP)\n{tp}", f"Missed (FN)\n{fn}", f"False Alarm (FP)\n{fp}"]
    colors = ["steelblue", "tomato", "darkorange"]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title(
        f"Lesion-Level Breakdown\n"
        f"(conf≥{totals['conf_thresh']}, IoU≥{totals['iou_thresh']})\n"
        f"Total GT lesions: {totals['total_gt_lesions']}",
        fontsize=10, fontweight="bold",
    )

    # ── Panel B: Per-image TP / FN / FP counts ────────────────────────────────
    ax = axes[1]
    img_ids    = np.arange(len(per_image))
    img_tp     = [r["tp"] for r in per_image]
    img_fn     = [r["fn"] for r in per_image]
    img_fp     = [r["fp"] for r in per_image]

    ax.bar(img_ids, img_tp, color="steelblue", alpha=0.85, label="Detected (TP)", width=1.0)
    ax.bar(img_ids, img_fn, bottom=img_tp, color="tomato", alpha=0.85,
           label="Missed (FN)", width=1.0)
    # FP shown below zero axis for readability
    ax.bar(img_ids, [-v for v in img_fp], color="darkorange", alpha=0.7,
           label="False Alarm (FP)", width=1.0)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Image index")
    ax.set_ylabel("Lesion count  (FP shown below 0)")
    ax.set_title("Per-Image Lesion Counts\n(TP/FN above zero · FP below zero)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # ── Panel C: Summary stats bar ────────────────────────────────────────────
    ax = axes[2]
    stat_labels = [
        "Detection Rate\n(TP/GT)",
        "Miss Rate\n(FN/GT)",
        "FA per Image\n(FP/N_imgs)",
    ]
    stat_values = [
        totals["detection_rate"],
        totals["miss_rate"],
        totals["false_alarm_per_img"],
    ]
    stat_colors = ["steelblue", "tomato", "darkorange"]
    bars = ax.bar(stat_labels, stat_values, color=stat_colors, width=0.55, alpha=0.9)
    for bar, v in zip(bars, stat_values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(stat_values) * 1.25 + 0.1)
    ax.set_ylabel("Rate")
    ax.set_title("Summary Statistics",
                 fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.suptitle(
        f"Test Set Lesion Detection Breakdown  "
        f"(conf≥{totals['conf_thresh']}, IoU≥{totals['iou_thresh']})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    _savefig(fig, out_dir, "plot_04_lesion_breakdown.png")


def _draw_boxes(ax, boxes, color, lw=2, ls="-", label=None, alpha=1.0):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=lw, edgecolor=color, linestyle=ls,
            facecolor="none", alpha=alpha,
            label=label if i == 0 else None,
        )
        ax.add_patch(rect)


def plot_predictions_grid(dataset, all_preds, all_targets, per_image,
                          out_dir: Path, n_cols: int = 4):
    """
    Grid of test images annotated with:
      green solid   — GT (detected / TP)
      green dashed  — GT (missed / FN)
      blue solid    — predicted TP
      red solid     — predicted FP (false alarm)

    Selects images from three categories:
      - Good detections (has GT, all detected, no FP)
      - Missed lesions  (has GT, at least one FN)
      - False alarms    (has FP, shown even if no GT)
    """
    n_rows = 3
    n_per_row = n_cols

    def _pick(condition, max_n):
        idxs = [i for i, r in enumerate(per_image) if condition(r)]
        if not idxs:
            return []
        step = max(1, len(idxs) // max_n)
        return idxs[::step][:max_n]

    good_idxs = _pick(lambda r: r["n_gt"] > 0 and r["fn"] == 0 and r["fp"] == 0, n_per_row)
    miss_idxs = _pick(lambda r: r["fn"] > 0, n_per_row)
    fp_idxs   = _pick(lambda r: r["fp"] > 0, n_per_row)

    rows = [
        ("Good detections  (FN=0, FP=0)", good_idxs),
        ("Missed lesions   (FN > 0)",      miss_idxs),
        ("False alarms     (FP > 0)",      fp_idxs),
    ]

    actual_rows = [(title, idxs) for title, idxs in rows if idxs]
    if not actual_rows:
        print("  No images to visualise — skipping prediction grid.")
        return

    n_r = len(actual_rows)
    fig, axes = plt.subplots(n_r, n_per_row,
                             figsize=(4 * n_per_row, 4.5 * n_r))
    if n_r == 1:
        axes = axes[None, :]
    if n_per_row == 1:
        axes = axes[:, None]

    for row_i, (title, idxs) in enumerate(actual_rows):
        for col_i in range(n_per_row):
            ax = axes[row_i, col_i]
            ax.axis("off")

            if col_i >= len(idxs):
                continue

            idx = idxs[col_i]
            image, _ = dataset[idx]
            br = per_image[idx]

            img_np = image.permute(1, 2, 0).cpu().numpy()
            ax.imshow(img_np, cmap="gray" if img_np.shape[2] == 1 else None)

            # GT boxes
            _draw_boxes(ax, br["gt_boxes_tp"], color="limegreen", lw=2,
                        ls="-",  label="GT (detected)")
            _draw_boxes(ax, br["gt_boxes_fn"], color="yellow",    lw=2,
                        ls="--", label="GT (missed)")
            # Predicted boxes
            _draw_boxes(ax, br["pred_boxes_tp"], color="royalblue", lw=2,
                        label="Pred TP")
            _draw_boxes(ax, br["pred_boxes_fp"], color="tomato",    lw=2,
                        label="Pred FP")

            tp = br["tp"]; fn = br["fn"]; fp = br["fp"]
            ax.set_title(f"img {idx}  TP={tp} FN={fn} FP={fp}", fontsize=8)

        axes[row_i, 0].set_ylabel(title, fontsize=9, fontweight="bold", rotation=90,
                                   labelpad=6)

    # Fixed legend — built manually so all 4 entries always appear regardless
    # of which box types happen to be present in the first plotted panel.
    legend_handles = [
        patches.Patch(edgecolor="limegreen",  facecolor="none", lw=2, ls="-",
                      label="GT lesion — detected (TP)"),
        patches.Patch(edgecolor="yellow",     facecolor="none", lw=2, ls="--",
                      label="GT lesion — missed (FN)"),
        patches.Patch(edgecolor="royalblue",  facecolor="none", lw=2, ls="-",
                      label="Prediction — true positive (TP)"),
        patches.Patch(edgecolor="tomato",     facecolor="none", lw=2, ls="-",
                      label="Prediction — false alarm (FP)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Test Set Predictions", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _savefig(fig, out_dir, "plot_05_predictions_grid.png")


# ══════════════════════════════════════════════════════════════════════════════
# Per-scan grouping and statistics
# ══════════════════════════════════════════════════════════════════════════════

def _scan_id(filename: str) -> str:
    """Extract scan ID from filename e.g. 'CEM005A1-01184.png' → 'CEM005A1'."""
    stem = Path(filename).stem          # 'CEM005A1-01184'
    dash = stem.rfind("-")
    return stem[:dash] if dash != -1 else stem


def group_by_scan(dataset, per_image: list) -> dict:
    """
    Group per-image results by scan ID.

    Returns
    -------
    dict  scan_id → {
        'slice_indices': list[int],       # dataset indices in order
        'filenames':     list[str],
        'slice_numbers': list[int],       # numeric part of filename
        'per_slice':     list[dict],      # per_image entries in order
        'n_gt':          int,             # total GT annotations across scan
        'tp':            int,
        'fn':            int,
        'fp':            int,
        'n_positive_slices':     int,     # slices with ≥1 GT box
        'n_detected_slices':     int,     # positive slices with ≥1 TP
        'detection_rate':        float,
        'miss_rate':             float,
        'false_alarms_per_slice':float,
    }
    """
    scans: dict = {}
    for idx, fname in enumerate(dataset.image_files):
        sid = _scan_id(fname)
        stem = Path(fname).stem
        try:
            slice_num = int(stem[stem.rfind("-") + 1:])
        except ValueError:
            slice_num = idx

        if sid not in scans:
            scans[sid] = {
                "slice_indices": [],
                "filenames":     [],
                "slice_numbers": [],
                "per_slice":     [],
            }
        scans[sid]["slice_indices"].append(idx)
        scans[sid]["filenames"].append(fname)
        scans[sid]["slice_numbers"].append(slice_num)
        scans[sid]["per_slice"].append(per_image[idx])

    # Compute aggregates
    for sid, info in scans.items():
        # Sort by slice number so the heatmap is in scan order
        order = np.argsort(info["slice_numbers"])
        info["slice_indices"] = [info["slice_indices"][i] for i in order]
        info["filenames"]     = [info["filenames"][i]     for i in order]
        info["slice_numbers"] = [info["slice_numbers"][i] for i in order]
        info["per_slice"]     = [info["per_slice"][i]     for i in order]

        n_slices = len(info["per_slice"])
        n_gt = sum(r["n_gt"] for r in info["per_slice"])
        tp   = sum(r["tp"]   for r in info["per_slice"])
        fn   = sum(r["fn"]   for r in info["per_slice"])
        fp   = sum(r["fp"]   for r in info["per_slice"])

        n_pos = sum(1 for r in info["per_slice"] if r["n_gt"] > 0)
        n_det = sum(1 for r in info["per_slice"] if r["n_gt"] > 0 and r["tp"] > 0)

        info.update({
            "n_slices":              n_slices,
            "n_gt":                  n_gt,
            "tp":                    tp,
            "fn":                    fn,
            "fp":                    fp,
            "n_positive_slices":     n_pos,
            "n_detected_slices":     n_det,
            "detection_rate":        tp / n_gt if n_gt > 0 else 0.0,
            "miss_rate":             fn / n_gt if n_gt > 0 else 0.0,
            "false_alarms_per_slice":fp / n_slices,
        })

    return dict(sorted(scans.items()))


def plot_per_scan_breakdown(scan_stats: dict, out_dir: Path):
    """
    Two-panel figure:
      Left  — stacked bar per scan (TP / FN / FP counts)
      Right — grouped bar of detection rate, miss rate, FA/slice per scan
    """
    scan_ids = list(scan_stats.keys())
    n = len(scan_ids)
    x = np.arange(n)

    tp_vals  = [scan_stats[s]["tp"]   for s in scan_ids]
    fn_vals  = [scan_stats[s]["fn"]   for s in scan_ids]
    fp_vals  = [scan_stats[s]["fp"]   for s in scan_ids]
    n_gt     = [scan_stats[s]["n_gt"] for s in scan_ids]
    det_rate = [scan_stats[s]["detection_rate"]        for s in scan_ids]
    miss_rate= [scan_stats[s]["miss_rate"]             for s in scan_ids]
    fa_rate  = [scan_stats[s]["false_alarms_per_slice"] for s in scan_ids]
    n_slices = [scan_stats[s]["n_slices"] for s in scan_ids]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: stacked count bar ───────────────────────────────────────────────
    ax = axes[0]
    b1 = ax.bar(x, tp_vals, color="steelblue",  alpha=0.9, label="Detected (TP)")
    b2 = ax.bar(x, fn_vals, bottom=tp_vals,      color="tomato",    alpha=0.9, label="Missed (FN)")
    # FP shown as negative bars below zero
    b3 = ax.bar(x, [-v for v in fp_vals], color="darkorange", alpha=0.8, label="False Alarm (FP)")
    ax.axhline(0, color="black", lw=0.8)

    # Annotate total GT and n_slices
    for i, (gt, ns) in enumerate(zip(n_gt, n_slices)):
        ax.text(i, max(tp_vals[i] + fn_vals[i] + 0.3, 0.5),
                f"GT={gt}\n({ns} sl.)", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(scan_ids, rotation=20, ha="right")
    ax.set_ylabel("Annotation count  (FP below 0)")
    ax.set_title("Per-Scan Annotation Counts\n(TP / FN above · FP below)",
                 fontweight="bold")
    ax.legend(fontsize=8)

    # ── Right: rate comparison ────────────────────────────────────────────────
    ax2 = axes[1]
    w = 0.25
    ax2.bar(x - w, det_rate,  width=w, color="steelblue",  alpha=0.9, label="Detection rate")
    ax2.bar(x,     miss_rate, width=w, color="tomato",     alpha=0.9, label="Miss rate")
    ax2.bar(x + w, fa_rate,   width=w, color="darkorange", alpha=0.8, label="FA / slice")

    for i, (d, m, f) in enumerate(zip(det_rate, miss_rate, fa_rate)):
        for val, offset in [(d, -w), (m, 0), (f, w)]:
            ax2.text(i + offset, val + 0.01, f"{val:.2f}",
                     ha="center", va="bottom", fontsize=7.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(scan_ids, rotation=20, ha="right")
    ax2.set_ylim(0, 1.25)
    ax2.set_ylabel("Rate")
    ax2.set_title("Per-Scan Detection / Miss / False-Alarm Rates",
                  fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)

    fig.suptitle(
        f"Per-Scan Lesion Detection Summary  "
        f"(conf≥{CONF_THRESH_FIXED}, IoU≥{IOU_THRESH})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    _savefig(fig, out_dir, "plot_06_per_scan_breakdown.png")


def plot_scan_slice_heatmap(scan_stats: dict, out_dir: Path):
    """
    One row per scan, one cell per slice.

    Colour encodes the combination of TP / FP for that slice:

      Positive slices (n_gt > 0):
        dark green  — detected (TP > 0), no false alarm  (FP = 0)
        teal        — detected (TP > 0), has false alarm (FP > 0)
        red         — missed   (TP = 0), no false alarm  (FP = 0)
        orange      — missed   (TP = 0), has false alarm (FP > 0)

      Negative slices (n_gt = 0):
        amber       — false alarm (FP > 0)
        light grey  — clean (FP = 0)

      White — padding (scan shorter than the longest scan)
    """
    scan_ids   = list(scan_stats.keys())
    max_slices = max(s["n_slices"] for s in scan_stats.values())

    DARK_GREEN  = np.array([0.13, 0.55, 0.13])   # detected, no FP
    TEAL        = np.array([0.00, 0.60, 0.60])   # detected + FP
    RED         = np.array([0.84, 0.19, 0.15])   # missed, no FP
    ORANGE      = np.array([0.99, 0.50, 0.05])   # missed + FP
    AMBER       = np.array([1.00, 0.75, 0.00])   # FP on background slice
    LIGHT_GREY  = np.array([0.90, 0.90, 0.90])   # clean background
    WHITE       = np.array([1.00, 1.00, 1.00])   # padding

    n_scans = len(scan_ids)
    rgb = np.ones((n_scans, max_slices, 3))       # white padding

    for row, sid in enumerate(scan_ids):
        for col, ps in enumerate(scan_stats[sid]["per_slice"]):
            if ps["n_gt"] == 0:
                rgb[row, col] = AMBER if ps["fp"] > 0 else LIGHT_GREY
            elif ps["tp"] > 0 and ps["fp"] == 0:
                rgb[row, col] = DARK_GREEN
            elif ps["tp"] > 0 and ps["fp"] > 0:
                rgb[row, col] = TEAL
            elif ps["tp"] == 0 and ps["fp"] > 0:
                rgb[row, col] = ORANGE
            else:                                 # tp=0, fp=0
                rgb[row, col] = RED

    fig_w = max(14, max_slices * 0.12)
    fig_h = max(3,  n_scans    * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(rgb, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(n_scans))
    ax.set_yticklabels(scan_ids, fontsize=10)
    ax.set_xlabel("Slice index (within scan, sorted by frame number)")
    ax.set_title("Per-Slice Detection Status Across Scans", fontsize=12, fontweight="bold")

    tick_step = 10
    ax.set_xticks(range(0, max_slices, tick_step))
    ax.set_xticklabels(range(0, max_slices, tick_step), fontsize=8)

    # Annotate each row with summary counts
    for row, sid in enumerate(scan_ids):
        s = scan_stats[sid]
        ax.text(max_slices + 1, row,
                f"GT={s['n_gt']}  TP={s['tp']}  FN={s['fn']}  FP={s['fp']}",
                va="center", ha="left", fontsize=8.5)

    ax.set_xlim(-0.5, max_slices + 1.5)

    import matplotlib.patches as mpatches
    legend_items = [
        mpatches.Patch(color=DARK_GREEN, label="Detected — no false alarm  (TP>0, FP=0)"),
        mpatches.Patch(color=TEAL,       label="Detected + false alarm     (TP>0, FP>0)"),
        mpatches.Patch(color=RED,        label="Missed — no false alarm    (TP=0, FP=0)"),
        mpatches.Patch(color=ORANGE,     label="Missed + false alarm       (TP=0, FP>0)"),
        mpatches.Patch(color=AMBER,      label="False alarm on background  (GT=0, FP>0)"),
        mpatches.Patch(color=LIGHT_GREY, label="Clean background           (GT=0, FP=0)"),
    ]
    ax.legend(handles=legend_items, loc="upper center",
              bbox_to_anchor=(0.42, -0.14), ncol=2, fontsize=8.5, frameon=True)

    fig.tight_layout(rect=[0, 0.10, 0.82, 1])
    _savefig(fig, out_dir, "plot_07_scan_slice_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Text report
# ══════════════════════════════════════════════════════════════════════════════

def write_text_report(metrics: dict, totals: dict, scan_stats: dict,
                      out_dir: Path, weights_path: str, split: str,
                      n_images: int, cfg: dict):
    lines = [
        "=" * 65,
        "  YOLOE Test Set Evaluation Report",
        "=" * 65,
        f"  Checkpoint  : {weights_path}",
        f"  Config      : {cfg.get('_config_path', 'n/a')}",
        f"  use_pu_loss : {cfg.get('use_pu_loss', False)}",
        f"  model_size  : {cfg.get('model_size', 's')}",
        f"  Split       : {split}  ({n_images} images)",
        f"  Conf thresh : {CONF_THRESH_FIXED} (fixed-threshold metrics)",
        f"  IoU thresh  : {IOU_THRESH}",
        "=" * 65,
        "",
        "── Standard Detection Metrics ──────────────────────────────",
        f"  AP@35              : {metrics.get('map_35',  0):.4f}",
        f"  AP@50              : {metrics.get('map_50',  0):.4f}",
        f"  AP@75              : {metrics.get('map_75',  0):.4f}",
        f"  mAP@50:95          : {metrics.get('map',     0):.4f}",
        "",
        "── Slice-Level Metrics ──────────────────────────────────────",
        f"  SliceAP@50         : {metrics.get('slice_ap_50', 0):.4f}",
        f"  SliceAP@75         : {metrics.get('slice_ap_75', 0):.4f}",
        "",
        f"── Fixed Threshold Metrics (conf ≥ {CONF_THRESH_FIXED}) ─────────────────",
        f"  Precision          : {metrics.get('precision', 0):.4f}",
        f"  Recall             : {metrics.get('recall',    0):.4f}",
        f"  F1                 : {metrics.get('f1',        0):.4f}",
        "",
        "── Operating Points ─────────────────────────────────────────",
        f"  P @ R=0.8          : {metrics.get('precision_at_recall80', 0):.4f}",
        f"  R @ P=0.8          : {metrics.get('recall_at_precision80', 0):.4f}",
        "",
        f"── Overall Lesion Breakdown (conf ≥ {totals['conf_thresh']}) ─────────────",
        f"  Total GT lesions   : {totals['total_gt_lesions']}",
        f"  Detected (TP)      : {totals['detected_tp']}",
        f"  Missed   (FN)      : {totals['missed_fn']}",
        f"  False alarms (FP)  : {totals['false_alarms_fp']}",
        f"  Detection rate     : {totals['detection_rate']:.4f}",
        f"  Miss rate          : {totals['miss_rate']:.4f}",
        f"  False alarms/image : {totals['false_alarm_per_img']:.4f}",
        "",
        "── Per-Scan Breakdown ───────────────────────────────────────",
        f"  {'Scan':<14} {'Slices':>6} {'GT':>5} {'TP':>5} {'FN':>5} {'FP':>5}  "
        f"{'DetRate':>8} {'MissRate':>9} {'FA/slice':>9}",
        "  " + "-" * 63,
    ]

    for sid, s in scan_stats.items():
        lines.append(
            f"  {sid:<14} {s['n_slices']:>6} {s['n_gt']:>5} "
            f"{s['tp']:>5} {s['fn']:>5} {s['fp']:>5}  "
            f"{s['detection_rate']:>8.4f} {s['miss_rate']:>9.4f} "
            f"{s['false_alarms_per_slice']:>9.4f}"
        )

    lines += [
        "  " + "-" * 63,
        "",
        "  Note: GT / TP / FN counts are at the annotation (box) level.",
        "  One 3-D lesion may appear as GT boxes in multiple consecutive slices.",
        "=" * 65,
    ]

    report_path = out_dir / "metrics_report.txt"
    text = "\n".join(lines)
    report_path.write_text(text)
    print(text)
    print(f"\n  Report saved: {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",  default="checkpoints/best_model_20260318_083558.pth")
    p.add_argument("--config",   default=None,
                   help="Path to training config YAML.  Reads model_size, "
                        "num_classes, use_pu_loss, and data_dir from it.  "
                        "CLI flags override config values where both are set.")
    p.add_argument("--data-dir", default=None,
                   help="Dataset root (overrides config data_dir if set)")
    p.add_argument("--split",    default="test",
                   help="Dataset split to evaluate on: train / val / test")
    p.add_argument("--batch-size",   type=int, default=None)
    p.add_argument("--num-workers",  type=int, default=4)
    p.add_argument("--n-cols",       type=int, default=4,
                   help="Columns in the prediction grid")
    p.add_argument("--conf-thresh", type=float, default=0.50,
                   help="Confidence threshold for fixed-threshold TP/FP/FN breakdown (default: 0.50)")
    p.add_argument("--iou-thresh",  type=float, default=0.50,
                   help="IoU threshold for TP/FP/FN matching (default: 0.50)")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: inference/test_results/)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load config YAML (if provided) ────────────────────────────────────────
    cfg: dict = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        cfg["_config_path"] = args.config
        print(f"Config    : {args.config}")
        print(f"  use_pu_loss : {cfg.get('use_pu_loss', False)}")
        print(f"  model_size  : {cfg.get('model_size', 's')}")

    # CLI flags override config where explicitly set
    data_dir    = args.data_dir   or cfg.get("data_dir",    "/projects/tenomix/ml-share/training/07/data")
    model_size  = cfg.get("model_size",  "s")
    num_classes = cfg.get("num_classes", 1)
    use_pu_loss = cfg.get("use_pu_loss", False)
    batch_size  = args.batch_size or cfg.get("batch_size", 16)

    # ── Thresholds (override module-level defaults) ────────────────────────────
    global CONF_THRESH_FIXED, IOU_THRESH
    CONF_THRESH_FIXED = args.conf_thresh
    IOU_THRESH        = args.iou_thresh
    print(f"Conf thresh (fixed): {CONF_THRESH_FIXED}")
    print(f"IoU  thresh        : {IOU_THRESH}")

    # ── Output directory ───────────────────────────────────────────────────────
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        script_dir = Path(__file__).resolve().parent
        suffix = "pu" if use_pu_loss else "no_pu"
        out_dir = script_dir / f"test_results_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # ── Dataset ────────────────────────────────────────────────────────────────
    # No augmentations: transform=None, mosaic_prob=0.0, copy_paste=None (defaults).
    # shuffle=False ensures images are evaluated in a deterministic order.
    dataset = UltrasoundDataset(root_dir=data_dir, split=args.split)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Dataset   : {args.split} split — {len(dataset)} images  (no augmentations)")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = load_model(args.weights, device,
                       model_size=model_size,
                       num_classes=num_classes,
                       use_pu_loss=use_pu_loss)

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"\nRunning inference on {len(dataset)} images ...")
    all_preds, all_targets, evaluator = run_inference(model, loader, device)
    print("  Done.")

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\nComputing metrics ...")
    metrics = evaluator.compute()

    # ── Lesion breakdown ──────────────────────────────────────────────────────
    per_image, totals = compute_lesion_breakdown(
        all_preds, all_targets,
        conf_thresh=CONF_THRESH_FIXED,
        iou_thresh=IOU_THRESH,
    )

    # ── Per-scan grouping ─────────────────────────────────────────────────────
    print("\nGrouping results by scan ...")
    scan_stats = group_by_scan(dataset, per_image)
    print(f"  Found {len(scan_stats)} scans: {list(scan_stats.keys())}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    scan_stats_json = {
        sid: {k: v for k, v in s.items()
              if k not in ("per_slice", "slice_indices", "filenames", "slice_numbers")}
        for sid, s in scan_stats.items()
    }
    combined = {"metrics": metrics, "lesion_breakdown": totals,
                "per_scan": scan_stats_json}
    json_path = out_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Metrics JSON: {json_path}")

    # ── Text report ────────────────────────────────────────────────────────────
    write_text_report(metrics, totals, scan_stats, out_dir,
                      args.weights, args.split, len(dataset), cfg)

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_metrics_summary(metrics, out_dir,
                         conf_thresh=CONF_THRESH_FIXED, iou_thresh=IOU_THRESH)
    plot_pr_curve(evaluator, metrics, out_dir,
                  conf_thresh=CONF_THRESH_FIXED, iou_thresh=IOU_THRESH)
    plot_score_distribution(evaluator, out_dir,
                            conf_thresh=CONF_THRESH_FIXED, iou_thresh=IOU_THRESH)
    plot_lesion_breakdown(totals, per_image, out_dir)
    plot_predictions_grid(dataset, all_preds, all_targets, per_image,
                          out_dir, n_cols=args.n_cols)
    plot_per_scan_breakdown(scan_stats, out_dir)
    plot_scan_slice_heatmap(scan_stats, out_dir)

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
