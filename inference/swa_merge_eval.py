#!/usr/bin/env python3
"""
SWA Merge + Eval Script
=======================

Overview
--------
After the Optuna hyperparameter search (tune_hyperparams_v4.py), each completed
trial saves its top-3 epoch checkpoints and a swa_manifest.yaml that lists them.
This script:

  1. Selects a trial  — either explicitly via --manifest, or automatically by
                        scanning train/config/hparam_v4_rank*.yaml and picking
                        the trial with the highest _optuna_slice_ap50 (--auto).

  2. Merges weights   — loads the model_state_dict from each listed checkpoint
                        and computes a uniform arithmetic mean over every
                        parameter tensor.  This is Stochastic Weight Averaging
                        (SWA, Izmailov et al. 2018): averaging nearby points in
                        weight space lands in a flatter loss basin and typically
                        generalises better than any single checkpoint.

  3. Recalibrates BN  — after averaging, each BatchNorm layer's running_mean /
                        running_var is the average of the individual checkpoints'
                        running stats, which are now mismatched to the merged
                        weights.  The script fixes this by running one full
                        forward pass over the training set in train() mode
                        (no gradients), which resets the running statistics to
                        values consistent with the merged weights.  This step
                        is ON by default; disable with --skip-bn-recalibration.

  4. Saves the model  — writes merged_model_<trial_number>.pt to the manifest
                        directory (or --output-dir).  The checkpoint includes
                        the recalibrated state_dict and provenance metadata
                        (source checkpoints, trial params, averaging method).

  5. Evaluates        — runs the merged model on the validation split and
                        reports AP@50, SliceAP@50, mAP, precision, recall, etc.
                        Results are also written to
                        merged_model_<trial_number>_metrics.json.

  6. Visualizes       — calls inference/visualize.py --merged-model, which
                        displays GT boxes (green) and merged-model predictions
                        (blue) for 6 random validation images and saves a PNG.


Model merging in detail
-----------------------
Each .pt checkpoint stores model_state_dict — the weights of the base YOLOE
model (i.e. YOLOEWithLoss.model).  The merge is a simple per-tensor mean:

    merged[k] = (w_epoch33[k] + w_epoch34[k] + w_epoch35[k]) / 3

All tensors — conv weights, biases, BN gamma/beta, and BN running stats — are
averaged in float32 regardless of their storage dtype.

Why the top-3 epochs of the best trial?
  The SWA paper shows that averaging a few consecutive near-peak checkpoints
  (rather than the single best) consistently improves generalisation.  The 3
  checkpoints are from a stable plateau of the same training run, so they share
  a loss basin; their average lies at the basin centre, which is usually flatter
  and more robust than any individual point.

Why BN recalibration is necessary?
  BN's running_mean / running_var are computed over the activations seen during
  training, not from the weights directly.  After weight averaging the activations
  change, so the old running stats no longer match.  A single no-gradient forward
  pass over the training set re-estimates them correctly for the merged weights.


Usage
-----
  # Quickest: auto-discover the best trial and run everything
  python inference/swa_merge_eval.py --auto

  # Point at a specific trial's manifest
  python inference/swa_merge_eval.py \\
      --manifest checkpoints/hparam_v4/trial_0014/swa_manifest.yaml

  # Custom search paths for --auto
  python inference/swa_merge_eval.py --auto \\
      --config-dir train/config \\
      --config-glob "hparam_v4_rank*.yaml" \\
      --checkpoint-dir checkpoints/hparam_v4

  # Save merged model elsewhere, skip visualization
  python inference/swa_merge_eval.py --auto \\
      --output-dir checkpoints/merged \\
      --no-visualize

  # Quick smoke-test (skip BN recalibration, skip visualization)
  python inference/swa_merge_eval.py --auto \\
      --skip-bn-recalibration --no-visualize


Outputs
-------
  checkpoints/hparam_v4/trial_XXXX/
    merged_model_<N>.pt               merged weights + BN stats + metadata
    merged_model_<N>_metrics.json     validation metrics as JSON

  inference/
    results_merged_merged_model_<N>.png   GT vs predictions grid (6 images)

Note
----
  Cross-trial weight averaging (merging weights from different Optuna trials)
  is intentionally not supported.  Models from different trials converge to
  different loss basins; averaging their weights causes destructive interference
  and collapses all metrics to zero.  Use per-trial SWA (--auto or --manifest)
  instead.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import glob
import io
import json
import subprocess

import yaml
import torch
from torch.utils.data import DataLoader

from model.yoloe import build_yoloe, YOLOEWithLoss
from data.dataset import UltrasoundDataset, collate_fn
from utils.metrics import decode_predictions, Evaluator

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="SWA weight averaging + validation + visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest', metavar='PATH',
                        help='Path to a specific swa_manifest.yaml (per-trial merge)')
    source.add_argument('--auto', action='store_true',
                        help='Auto-discover the best trial and do a per-trial merge')

    # Auto-discovery options
    p.add_argument('--config-dir', default=None,
                   help='Directory to search for ranked YAML configs '
                        '(default: <project_root>/train/config)')
    p.add_argument('--config-glob', default='hparam_v4_rank*.yaml',
                   help='Glob pattern for ranked config files '
                        '(default: hparam_v4_rank*.yaml)')
    p.add_argument('--checkpoint-dir', default=None,
                   help='Directory containing trial_XXXX sub-dirs '
                        '(default: <project_root>/checkpoints/hparam_v4)')

    # Evaluation options
    p.add_argument('--data-dir', default='/projects/tenomix/ml-share/training/07/data',
                   help='Dataset root directory')
    p.add_argument('--output-dir', default=None,
                   help='Where to save merged model + metrics '
                        '(defaults to the manifest directory)')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    # Merging options
    p.add_argument('--merge-top-k', type=int, default=None, metavar='K',
                   help='Use only the top-K checkpoints by ema_slice_ap50 for merging. '
                        'Default: use all checkpoints listed in the manifest. '
                        'Example: --merge-top-k 2 drops the weakest checkpoint.')
    p.add_argument('--averaging', choices=['uniform', 'weighted'], default='uniform',
                   help='Weight scheme for averaging checkpoints. '
                        '"uniform" = equal weights (default). '
                        '"weighted" = weight each checkpoint by its ema_slice_ap50 score, '
                        'so stronger checkpoints contribute more.')
    p.add_argument('--bn-recal-passes', type=int, default=1,
                   help='Number of full passes over the training set for BN recalibration '
                        '(default: 1). Increase to 2 if BN stats appear unstable.')
    p.add_argument('--skip-bn-recalibration', action='store_true',
                   help='Skip BatchNorm recalibration after merging (not recommended)')
    p.add_argument('--no-visualize', action='store_true',
                   help='Skip running visualize.py after evaluation')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def find_best_manifest(config_glob: str, config_dir: str, checkpoint_dir: str) -> str:
    """
    Scan ranked YAML config files, pick the trial with the highest
    _optuna_slice_ap50, and return the path to its swa_manifest.yaml.
    """
    pattern = os.path.join(config_dir, config_glob)
    config_files = glob.glob(pattern)
    if not config_files:
        raise FileNotFoundError(
            f"No ranked config files found matching: {pattern}"
        )

    best_score = -1.0
    best_trial = None
    best_config_path = None

    print(f"Scanning {len(config_files)} ranked config file(s) in: {config_dir}")
    for cfg_path in sorted(config_files):
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        score = cfg.get('_optuna_slice_ap50')
        trial = cfg.get('_optuna_trial_number')
        if score is None or trial is None:
            continue
        print(f"  {os.path.basename(cfg_path):50s}  trial={trial:4d}  slice_ap50={score:.4f}")
        if score > best_score:
            best_score = score
            best_trial = trial
            best_config_path = cfg_path

    if best_trial is None:
        raise ValueError(
            "No config files contained _optuna_trial_number / _optuna_slice_ap50 keys."
        )

    print(f"\nBest trial: {best_trial}  (EMA SliceAP@50 = {best_score:.4f})")
    print(f"  Source config: {best_config_path}")

    manifest_path = os.path.join(
        checkpoint_dir, f"trial_{best_trial:04d}", "swa_manifest.yaml"
    )
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Expected manifest not found: {manifest_path}\n"
            f"Make sure checkpoints for trial {best_trial} are present."
        )
    return manifest_path


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str) -> dict:
    with open(manifest_path, 'r') as f:
        return yaml.safe_load(f)


def average_checkpoints(checkpoint_paths: list, weights: list, device) -> dict:
    """
    Weighted average of model_state_dicts across K checkpoint files.

    For every named tensor k:
        merged[k] = sum_i( w_i * checkpoint_i[k] )   where sum(w_i) == 1.

    weights: normalised floats summing to 1.0 (pass [1/K]*K for uniform).
    All arithmetic is done in float32 regardless of stored dtype.
    """
    assert len(checkpoint_paths) == len(weights), "paths and weights must be same length"
    assert abs(sum(weights) - 1.0) < 1e-6, "weights must sum to 1"

    avg_state = None
    for path, w in zip(checkpoint_paths, weights):
        print(f"    Loading (weight={w:.4f}): {path}")
        ck = torch.load(path, map_location=device, weights_only=False)
        state = ck['model_state_dict']
        if avg_state is None:
            avg_state = {k: v.float().clone() * w for k, v in state.items()}
        else:
            for k in avg_state:
                avg_state[k].add_(state[k].float() * w)
    return avg_state


def recalibrate_bn(model, data_dir: str, batch_size: int, num_workers: int,
                   device, n_passes: int = 1):
    """
    Re-run the training set through the merged model in train() mode so that
    BatchNorm running_mean / running_var reflect the merged weights rather than
    the stale averaged statistics from the individual checkpoints.

    n_passes: number of full passes over the training set (default 1).
    Increase to 2 if BN stats appear unstable (small dataset, large batch).
    """
    train_dataset = UltrasoundDataset(root_dir=data_dir, split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
    )
    print(f"  BN recalibration: {len(train_dataset)} images × {n_passes} pass(es), "
          f"{len(train_loader)} batches/pass...")

    model.train()
    with torch.no_grad():
        for _ in range(n_passes):
            for images, _ in train_loader:
                model.model(images.to(device))

    print("  BN recalibration done.")


def run_validation(model, data_dir: str, batch_size: int, num_workers: int, device) -> dict:
    val_dataset = UltrasoundDataset(root_dir=data_dir, split='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
    )
    print(f"  Validation set size: {len(val_dataset)} images")

    evaluator = Evaluator()
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            device_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            cls_scores, reg_dists = model(images)

            h, w = images.shape[2:]
            feat_sizes = [(h // s, w // s) for s in model.model.strides]
            anchor_points, stride_tensor = model.model.get_anchor_points(
                feat_sizes, device, images.dtype
            )
            batch_preds = decode_predictions(cls_scores, reg_dists, anchor_points, stride_tensor)
            evaluator.update(batch_preds, device_targets)

    return evaluator.compute()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    'slice_ap_50': 'SliceAP@50',
    'precision':   'Precision  (IoU≥0.5, conf≥0.5)',
    'recall':      'Recall     (IoU≥0.5, conf≥0.5)',
    'f1':          'F1 Score   (IoU≥0.5, conf≥0.5)',
    'map_50':      'AP@50',
    'map_75':      'AP@75',
    'map':         'mAP (IoU 0.50:0.95)',
    'mar_1':       'AR@1',
    'mar_10':      'AR@10',
    'mar_100':     'Recall@100',
}


def build_model_from_checkpoint(checkpoint_path: str, device) -> 'YOLOEWithLoss':
    """Load a single .pt checkpoint into a fresh model instance."""
    base_model = build_yoloe(model_size='s', num_classes=1)
    model = YOLOEWithLoss(model=base_model, num_classes=1).to(device)
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.model.load_state_dict(ck['model_state_dict'])
    return model


def print_comparison(best_ck: dict, metrics_best: dict,
                     metrics_merged: dict, merge_label: str,
                     averaging: str = 'uniform', weights: list = None):
    """Print a side-by-side comparison table of best-single vs SWA-merged metrics."""
    KEY_ORDER = ['slice_ap_50', 'precision', 'recall', 'f1', 'map_50', 'map_75', 'map', 'mar_100']
    all_keys = KEY_ORDER + [k for k in sorted(metrics_best) if k not in KEY_ORDER]

    col_w = 10
    label_w = 38

    header = (f"\n{'Metric':<{label_w}}  "
              f"{'Best single':>{col_w}}  "
              f"{'SWA merged':>{col_w}}  "
              f"{'Delta':>{col_w}}")
    sep = "-" * len(header)

    weight_str = (f"  weights=[{', '.join(f'{w:.3f}' for w in weights)}]"
                  if averaging == 'weighted' and weights else "")
    best_trial = best_ck.get('trial_number', '?')
    print(f"\n{'='*len(header)}")
    print(f"  Best single: trial={best_trial} epoch={best_ck['epoch']} "
          f"(ema={best_ck['ema_slice_ap50']:.4f})  |  "
          f"Merged: {merge_label}  [{averaging}{weight_str}]")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for key in all_keys:
        if key not in metrics_best or key not in metrics_merged:
            continue
        label = METRIC_LABELS.get(key, key)
        v_best = metrics_best[key]
        v_merged = metrics_merged[key]
        delta = v_merged - v_best
        delta_str = f"{delta:+.4f}"
        # Highlight improvements
        marker = " ▲" if delta > 0.001 else (" ▼" if delta < -0.001 else "  ")
        print(f"  {label:<{label_w}}"
              f"  {v_best:{col_w}.4f}"
              f"  {v_merged:{col_w}.4f}"
              f"  {delta_str:>{col_w}}{marker}")

    print(sep)


def _compute_weights(merge_ckpts: list, averaging: str) -> list:
    if averaging == 'weighted':
        scores = [ck['ema_slice_ap50'] for ck in merge_ckpts]
        total = sum(scores)
        return [s / total for s in scores]
    return [1.0 / len(merge_ckpts)] * len(merge_ckpts)


def _run_merge_eval(args, device, merge_ckpts: list, best_ck: dict,
                    output_dir: str, merge_label: str, averaging_tag: str):
    """
    Shared pipeline: average checkpoints → BN recal → save → eval → compare → visualize.
    Called by both per-trial and cross-trial branches.
    """
    weights = _compute_weights(merge_ckpts, args.averaging)

    print(f"\nAveraging scheme: {args.averaging}  ({len(merge_ckpts)} checkpoints)")
    for ck, w in zip(merge_ckpts, weights):
        print(f"  trial={ck.get('trial_number','?'):4}  epoch={ck['epoch']:3d}  "
              f"ema={ck['ema_slice_ap50']:.4f}  weight={w:.4f}")

    # ── Evaluate best single ───────────────────────────────────────────────
    print(f"\n[1/2] Evaluating best single checkpoint "
          f"(trial={best_ck.get('trial_number','?')} epoch={best_ck['epoch']} "
          f"ema={best_ck['ema_slice_ap50']:.4f})...")
    model_best = build_model_from_checkpoint(best_ck['path'], device)
    metrics_best = run_validation(
        model_best, args.data_dir, args.batch_size, args.num_workers, device
    )
    del model_best

    # ── Build merged model ─────────────────────────────────────────────────
    print(f"\n[2/2] Building merged model ({merge_label})...")
    avg_state = average_checkpoints(
        [ck['path'] for ck in merge_ckpts], weights=weights, device=device
    )
    base_model = build_yoloe(model_size='s', num_classes=1)
    model_merged = YOLOEWithLoss(model=base_model, num_classes=1).to(device)
    model_merged.model.load_state_dict(avg_state)

    if not args.skip_bn_recalibration:
        print("\nRecalibrating BatchNorm running statistics...")
        recalibrate_bn(model_merged, args.data_dir, args.batch_size, args.num_workers,
                       device, n_passes=args.bn_recal_passes)
    else:
        print("\nSkipping BN recalibration (--skip-bn-recalibration set).")

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    merged_path = os.path.join(output_dir, f"merged_{averaging_tag}.pt")
    buf = io.BytesIO()
    torch.save({
        'model_state_dict': model_merged.model.state_dict(),
        'merge_label':      merge_label,
        'averaging':        args.averaging,
        'merge_weights':    weights,
        'source_checkpoints': [ck['path'] for ck in merge_ckpts],
        'bn_recalibrated':  not args.skip_bn_recalibration,
        'bn_recal_passes':  args.bn_recal_passes if not args.skip_bn_recalibration else 0,
    }, buf)
    with open(merged_path, 'wb') as f:
        f.write(buf.getbuffer())
    print(f"\nSaved merged model → {merged_path}")

    # ── Evaluate merged ────────────────────────────────────────────────────
    print("\nEvaluating merged model...")
    metrics_merged = run_validation(
        model_merged, args.data_dir, args.batch_size, args.num_workers, device
    )

    # ── Comparison table ───────────────────────────────────────────────────
    print_comparison(best_ck, metrics_best, metrics_merged,
                     merge_label=merge_label,
                     averaging=args.averaging, weights=weights)

    # ── Save metrics JSON ──────────────────────────────────────────────────
    metrics_path = os.path.join(output_dir, f"merged_{averaging_tag}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            'merge_label': merge_label,
            'best_single': {
                'path':              best_ck['path'],
                'epoch':             best_ck['epoch'],
                'trial_number':      best_ck.get('trial_number'),
                'ema_slice_ap50':    best_ck['ema_slice_ap50'],
                'metrics':           metrics_best,
            },
            'swa_merged': {
                'path':       merged_path,
                'averaging':  args.averaging,
                'source_checkpoints': [
                    {'path': ck['path'], 'epoch': ck['epoch'],
                     'trial_number': ck.get('trial_number'),
                     'ema_slice_ap50': ck['ema_slice_ap50'], 'weight': w}
                    for ck, w in zip(merge_ckpts, weights)
                ],
                'metrics': metrics_merged,
            },
        }, f, indent=2)
    print(f"Metrics saved → {metrics_path}")

    # ── Visualize ──────────────────────────────────────────────────────────
    if not args.no_visualize:
        print("\nLaunching visualize.py...")
        visualize_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualize.py')
        subprocess.run([
            sys.executable, visualize_script,
            '--best-model',   best_ck['path'],
            '--merged-model', merged_path,
            '--data-dir',     args.data_dir,
        ], check=True)


def main():
    args = parse_args()
    device = torch.device(args.device)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    checkpoint_dir = args.checkpoint_dir or os.path.join(
        project_root, 'checkpoints', 'hparam_v4'
    )
    output_dir = args.output_dir or os.path.join(project_root, 'checkpoints', 'merged')

    if args.auto:
        config_dir = args.config_dir or os.path.join(project_root, 'train', 'config')
        manifest_path = find_best_manifest(args.config_glob, config_dir, checkpoint_dir)
    else:
        manifest_path = os.path.abspath(args.manifest)

    print(f"\nLoading manifest: {manifest_path}")
    manifest = load_manifest(manifest_path)
    trial_number = manifest['trial_number']
    output_dir = args.output_dir or os.path.dirname(manifest_path)

    checkpoints_sorted = sorted(manifest['checkpoints'],
                                key=lambda c: c['ema_slice_ap50'], reverse=True)
    top_k = args.merge_top_k or len(checkpoints_sorted)
    merge_ckpts = checkpoints_sorted[:top_k]

    # Attach trial_number so _run_merge_eval can display it
    for ck in merge_ckpts:
        ck.setdefault('trial_number', trial_number)

    print(f"Trial {trial_number} — {len(manifest['checkpoints'])} checkpoints "
          f"(using top-{top_k}):")
    for ck in checkpoints_sorted:
        tag = '[merge]' if ck in merge_ckpts else '[skip] '
        print(f"  {tag} epoch={ck['epoch']:3d}  "
              f"slice_ap50={ck['slice_ap50']:.4f}  "
              f"ema_slice_ap50={ck['ema_slice_ap50']:.4f}")

    best_ck = checkpoints_sorted[0]
    best_ck.setdefault('trial_number', trial_number)
    merge_label = f"per-trial {trial_number}, top-{top_k}"
    averaging_tag = f"trial{trial_number}_{args.averaging}_top{top_k}"

    _run_merge_eval(args, device, merge_ckpts, best_ck,
                    output_dir, merge_label, averaging_tag)


if __name__ == '__main__':
    main()
