"""
Validation diagnostic plots for object detection training runs.

Six-panel figure saved to results_dir at end of training:
  1. PR curve (IoU=0.5)          — with AP area, P@R=0.8 and R@P=0.8 marked
  2. F1 / Precision / Recall     — vs confidence threshold (swept from PR curve)
  3. Score distribution          — TP vs FP confidence histogram
  4. Train & val loss            — over epochs
  5. mAP family                  — mAP@35/50/75, mAP@50:95, SliceAP@50/75 over epochs
  6. P / R / F1 / P@R=0.8        — scalar metrics over epochs
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive; safe on headless servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def plot_validation_summary(evaluator, training_info, save_path):
    """
    Generate a 6-panel validation diagnostic figure and save as PNG.

    Args:
        evaluator:      Evaluator instance whose _all_preds/_all_targets are still
                        populated (i.e. called right after validate() returns).
        training_info:  The training_info dict from train.py (metrics lists per epoch).
        save_path:      Destination .png path.
    """
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    _plot_pr_curve(fig.add_subplot(gs[0, 0]), evaluator)
    _plot_f1_vs_threshold(fig.add_subplot(gs[0, 1]), evaluator)
    _plot_score_distribution(fig.add_subplot(gs[0, 2]), evaluator)
    _plot_loss_curves(fig.add_subplot(gs[1, 0]), training_info)
    _plot_map_curves(fig.add_subplot(gs[1, 1]), training_info)
    _plot_prf1_curves(fig.add_subplot(gs[1, 2]), training_info)

    best_ep = training_info.get("best_epoch", "?")
    best_ap = training_info.get("best_ap_50", float("nan"))
    fig.suptitle(
        f'Validation Diagnostics  |  Best epoch: {best_ep}  |  Best AP@50: {best_ap:.4f}',
        fontsize=13, fontweight='bold', y=1.01
    )

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved validation plots → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Panel helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_pr_curve(ax, evaluator):
    """Panel 1: Precision-Recall curve at IoU=0.5."""
    prec, rec, scores = evaluator.pr_curve_data(iou_thresh=0.5)

    # Sort by recall for a clean curve
    order = np.argsort(rec)
    rec_s, prec_s = rec[order], prec[order]

    # Area under envelope (AP)
    ap = float(np.trapezoid(prec_s, rec_s)) if len(rec_s) > 1 else 0.0

    ax.fill_between(rec_s, prec_s, alpha=0.15, color='steelblue')
    ax.plot(rec_s, prec_s, color='steelblue', lw=2, label=f'AP@50 = {ap:.3f}')

    # Mark P@R=0.8
    p_at_r80 = max((p for p, r in zip(prec, rec) if r >= 0.8), default=None)
    if p_at_r80 is not None:
        ax.axvline(0.8, color='tomato', lw=1, ls='--', alpha=0.7)
        ax.scatter([0.8], [p_at_r80], color='tomato', zorder=5,
                   label=f'P@R=0.8 = {p_at_r80:.3f}')

    # Mark R@P=0.8
    r_at_p80 = max((r for p, r in zip(prec, rec) if p >= 0.8), default=None)
    if r_at_p80 is not None:
        ax.axhline(0.8, color='darkorange', lw=1, ls='--', alpha=0.7)
        ax.scatter([r_at_p80], [0.8], color='darkorange', zorder=5,
                   label=f'R@P=0.8 = {r_at_p80:.3f}')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('PR Curve (IoU = 0.5)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)


def _plot_f1_vs_threshold(ax, evaluator):
    """Panel 2: Precision / Recall / F1 vs confidence threshold."""
    prec, rec, scores = evaluator.pr_curve_data(iou_thresh=0.5)

    if len(scores) < 2:
        ax.text(0.5, 0.5, 'Not enough predictions', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('F1 vs Threshold')
        return

    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)

    # Scores decrease along the curve; reverse so x-axis goes low → high threshold
    thresholds = scores[::-1]
    prec_r = prec[::-1]
    rec_r = rec[::-1]
    f1_r = f1[::-1]

    ax.plot(thresholds, prec_r, color='steelblue', lw=1.5, label='Precision')
    ax.plot(thresholds, rec_r, color='seagreen', lw=1.5, label='Recall')
    ax.plot(thresholds, f1_r, color='darkorange', lw=2, label='F1')

    # Mark optimal F1 threshold
    best_idx = int(np.argmax(f1_r))
    best_thresh = thresholds[best_idx]
    best_f1 = f1_r[best_idx]
    ax.axvline(best_thresh, color='darkorange', lw=1, ls='--', alpha=0.7)
    ax.scatter([best_thresh], [best_f1], color='darkorange', zorder=5,
               label=f'Best F1={best_f1:.3f} @ {best_thresh:.2f}')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Confidence threshold')
    ax.set_ylabel('Metric value')
    ax.set_title('P / R / F1 vs Confidence Threshold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_score_distribution(ax, evaluator):
    """Panel 3: Histogram of prediction scores split by TP / FP."""
    tp_scores, fp_scores = evaluator.score_distributions(iou_thresh=0.5)

    bins = np.linspace(0, 1, 31)
    if tp_scores:
        ax.hist(tp_scores, bins=bins, color='steelblue', alpha=0.65,
                label=f'TP  (n={len(tp_scores)})', density=True)
    if fp_scores:
        ax.hist(fp_scores, bins=bins, color='tomato', alpha=0.65,
                label=f'FP  (n={len(fp_scores)})', density=True)

    ax.set_xlabel('Confidence score')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution (IoU = 0.5)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_loss_curves(ax, training_info):
    """Panel 4: Train / val loss over epochs."""
    m = training_info['metrics']
    epochs = range(1, len(m['train_loss']) + 1)

    ax.plot(epochs, m['train_loss'], color='steelblue', lw=1.5, label='Train loss')
    ax.plot(epochs, m['val_loss'], color='tomato', lw=1.5, label='Val loss')

    best_ep = training_info.get('best_epoch', None)
    if best_ep and best_ep > 0:
        ax.axvline(best_ep, color='gray', lw=1, ls='--', alpha=0.6, label=f'Best epoch {best_ep}')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_map_curves(ax, training_info):
    """Panel 5: mAP family metrics over epochs."""
    m = training_info['metrics']
    epochs = range(1, len(m['map_50']) + 1)

    series = [
        ('map_35',     'mAP@35',     'mediumpurple', '-'),
        ('map_50',     'mAP@50',     'steelblue',    '-'),
        ('map_75',     'mAP@75',     'seagreen',     '-'),
        ('map',        'mAP@50:95',  'darkorange',   '-'),
        ('slice_ap_50','SliceAP@50', 'steelblue',    '--'),
        ('slice_ap_75','SliceAP@75', 'seagreen',     '--'),
    ]
    for key, label, color, ls in series:
        if key in m and m[key]:
            ax.plot(epochs, m[key], color=color, lw=1.5, ls=ls, label=label)

    best_ep = training_info.get('best_epoch', None)
    if best_ep and best_ep > 0:
        ax.axvline(best_ep, color='gray', lw=1, ls=':', alpha=0.6)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('AP')
    ax.set_title('mAP & SliceAP over Epochs')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)


def _plot_prf1_curves(ax, training_info):
    """Panel 6: Precision / Recall / F1 and P@R=0.8 over epochs."""
    m = training_info['metrics']
    epochs = range(1, len(m['precision']) + 1)

    series = [
        ('precision',           'Precision',  'steelblue',  '-'),
        ('recall',              'Recall',     'seagreen',   '-'),
        ('f1',                  'F1',         'darkorange', '-'),
        ('precision_at_recall80', 'P@R=0.8', 'tomato',     '--'),
        ('recall_at_precision80', 'R@P=0.8', 'mediumpurple','--'),
    ]
    for key, label, color, ls in series:
        if key in m and m[key]:
            ax.plot(epochs, m[key], color=color, lw=1.5, ls=ls, label=label)

    best_ep = training_info.get('best_epoch', None)
    if best_ep and best_ep > 0:
        ax.axvline(best_ep, color='gray', lw=1, ls=':', alpha=0.6)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('P / R / F1 / Operating Points over Epochs')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
