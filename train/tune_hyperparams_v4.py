"""
YOLOE Hyperparameter Tuning v4 with Optuna.

═══════════════════════════════════════════════════════════════════════════════
Changes from v3
═══════════════════════════════════════════════════════════════════════════════

Metric changes
──────────────
• Objective changed from raw AP@50 → SliceAP@50 (slice_ap_50 from the
  Evaluator).  SliceAP@50 treats each of the 357 validation slices as one
  binary sample (positive = has at least one GT box; hit = any prediction
  overlaps any GT at IoU ≥ 0.5).  With 357 samples it is substantially more
  stable than box-level AP@50, whose effective sample count is the number of
  GT boxes (typically far fewer) — see commit notes for full analysis.

• EMA smoothing (α=0.3) is applied to SliceAP@50 for early stopping and for
  the value returned to Optuna as the trial objective.  The EMA is NOT
  reported to the pruner — the pruner sees raw per-epoch SliceAP@50 so that
  rung comparisons reflect the model's actual state at that epoch rather than
  a lagged average.

  EMA update rule (applied after every epoch):
      ema_t = α * slice_ap_t  +  (1 − α) * ema_{t-1},   ema_0 = slice_ap_1

  With α=0.3 the effective half-life is ~2 epochs, smoothing out the ±0.1
  single-epoch swings observed in v3 logs while still tracking real trends.

Pruner change
─────────────
• MedianPruner replaced by HyperbandPruner.
  - MedianPruner compares a trial's AP against the global median at the same
    epoch; this was pruning good configs that hit a bad checkpoint at the
    comparison epoch.
  - HyperbandPruner (successive halving in brackets) only compares trials
    within the same bracket, which are started with similar budgets.  It is
    more robust to the early-epoch noise in this dataset.
  - Configuration: min_resource=10 (no pruning before epoch 10, past warmup),
    max_resource=50 (matches --epochs), reduction_factor=3 (rungs at 10, 30,
    50; keeps top ⅓ at each rung).

Checkpoint saving (SWA preparation)
────────────────────────────────────
• Each trial now saves the top-K model checkpoints by EMA SliceAP@50
  (default K=3).  These are saved to:
      <checkpoint_dir>/trial_<N>/epoch<E>_sliceap<V>.pt
  and a manifest is written at the end of the trial to:
      <checkpoint_dir>/trial_<N>/swa_manifest.yaml

  The manifest lists each checkpoint's path, epoch, raw SliceAP@50, and EMA
  SliceAP@50.  To perform SWA weight averaging after the study completes,
  load the top-trial manifests and average the listed state_dicts.

  Checkpoint format (keys saved in each .pt file):
      model_state_dict  — base YOLOE model weights (YOLOEWithLoss.model),
                          directly loadable by build_yoloe() for inference.
      epoch             — epoch number within this trial.
      slice_ap50        — raw SliceAP@50 at this epoch.
      ema_slice_ap50    — EMA SliceAP@50 at this epoch (used for selection).
      trial_number      — Optuna trial number.
      trial_params      — full dict of hyperparameters for this trial.

  Disk budget: ~300 trials × 3 checkpoints × ~25 MB (YOLOE-s) ≈ 22 GB on
  NFS.  Checkpoints for pruned trials are deleted immediately on pruning to
  keep disk usage bounded; only completed-trial checkpoints are retained.

Output
──────
• Best config saved as best_hyperparams_v4.yaml (was v3).
• Per-trial ranked configs saved as hparam_v4_rank<N>_sliceap<V>.yaml.
• Optuna objective and YAML metric key renamed from _optuna_ap50 to
  _optuna_slice_ap50 to reflect the change in objective.

═══════════════════════════════════════════════════════════════════════════════
Search space (unchanged from v3)
═══════════════════════════════════════════════════════════════════════════════
Continuous:   learning_rate [1e-4, 5e-3 log], weight_decay [1e-4, 1e-2 log],
              mosaic_prob [0.1, 0.5], label_smooth [0.0, 0.15],
              copy_paste_p [0.0, 0.7]  (0.0 disables copy-paste)
Categorical:  slice_stride {5, 7, 10}, batch_size {8, 16, 32},
              use_clahe {T/F}, use_shift_scale_rotate {T/F},
              use_brightness_contrast {T/F},
              use_median_blur {T/F}, median_blur_limit {3, 5, 7},
              use_speckle_noise {T/F}, speckle_level {light, medium, heavy}
Fixed:        HorizontalFlip, VerticalFlip always on.
Removed:      GaussianBlur, GaussNoise (replaced by MedianBlur + speckle).

═══════════════════════════════════════════════════════════════════════════════
Usage
═══════════════════════════════════════════════════════════════════════════════
Single node:
    python train/tune_hyperparams_v4.py --config train/config/default.yaml \\
        --n-trials 40

Multi-node via SLURM array:
    sbatch slurm-script/tune_hyperparams_v4.sh

SWA weight averaging after the study (example):
    import torch, yaml, glob
    from model.yoloe import build_yoloe

    manifests = sorted(glob.glob("checkpoints/hparam_v4/trial_*/swa_manifest.yaml"))
    # Pick the manifests for your top trials (by _optuna_slice_ap50 in YAML configs)
    state_dicts = []
    for manifest_path in manifests[:3]:
        m = yaml.safe_load(open(manifest_path))
        for ckpt in m["checkpoints"]:
            sd = torch.load(ckpt["path"], map_location="cpu")["model_state_dict"]
            state_dicts.append(sd)
    # Average
    avg = {k: sum(sd[k] for sd in state_dicts) / len(state_dicts) for k in state_dicts[0]}
    model = build_yoloe(model_size="s", num_classes=1)
    model.load_state_dict(avg)
"""

import heapq
import os
import sys
import argparse
import time
import yaml

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.yoloe import build_yoloe, YOLOEWithLoss
from data.dataset import UltrasoundDataset, collate_fn
from data.sampler import ScanAwareSampler
from data.augmentations import TissueAwareCopyPaste
from train import train_one_epoch, validate, set_seed, seed_worker

# Augmentations dropped from the base config in v3+ (replaced by MedianBlur + speckle)
_REMOVED_AUGS = {"GaussianBlur", "GaussNoise"}

# Mapping from speckle_level trial value → MultiplicativeNoise multiplier range
_SPECKLE_LEVELS = {
    "light":  [0.95, 1.05],
    "medium": [0.90, 1.10],
    "heavy":  [0.80, 1.20],
}

# Trial parameter flags that are internal to the tuner and must not appear in
# the saved YAML config (they are expanded into augmentation/copy_paste blocks)
_TRIAL_FLAGS = {
    "use_clahe", "use_shift_scale_rotate", "use_brightness_contrast",
    "use_median_blur", "median_blur_limit",
    "use_speckle_noise", "speckle_level",
    "copy_paste_p",
}


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOE Hyperparameter Tuning v4 — SliceAP@50 objective, "
                    "EMA early stopping, HyperbandPruner, top-K checkpoint saving"
    )
    parser.add_argument(
        "--config", type=str, default="train/config/default.yaml",
        help="Base config YAML.  Data path, model size, and fixed training "
             "settings are read from here; tunable values are overridden per trial."
    )
    parser.add_argument(
        "--study-name", type=str, default="yoloe_hparam_search_v4",
        help="Optuna study name — shared across all parallel SLURM workers."
    )
    parser.add_argument(
        "--storage", type=str, default=None,
        help="Optuna storage URL.  Defaults to "
             "sqlite:///<project_root>/optuna_studies/<study_name>.db"
    )
    parser.add_argument(
        "--n-trials", type=int, default=5,
        help="Number of trials this worker will run."
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Maximum epochs per trial.  Early stopping usually cuts this short."
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=15,
        help="Epochs without EMA SliceAP@50 improvement before a trial is "
             "stopped early.  Counted against the EMA, not the raw metric, to "
             "avoid stopping on transient bad checkpoints."
    )
    parser.add_argument(
        "--early-stopping-min-delta", type=float, default=0.005,
        help="Minimum EMA SliceAP@50 improvement required to reset the "
             "early-stopping counter."
    )
    parser.add_argument(
        "--ema-alpha", type=float, default=0.3,
        help="EMA smoothing factor α for SliceAP@50.  Higher = less smoothing. "
             "α=0.3 gives an effective half-life of ~2 epochs."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed.  Each trial uses seed + trial.number for "
             "reproducibility while keeping trials independent."
    )
    parser.add_argument(
        "--save-top-k", type=int, default=5,
        help="Number of top completed trials to write as ranked YAML configs."
    )
    parser.add_argument(
        "--output-config-dir", type=str, default="train/config",
        help="Directory where ranked YAML configs and best_hyperparams_v4.yaml "
             "are written."
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints/hparam_v4",
        help="Root directory for per-trial checkpoint folders.  Each trial "
             "writes up to --ckpt-top-k .pt files and one swa_manifest.yaml here."
    )
    parser.add_argument(
        "--ckpt-top-k", type=int, default=3,
        help="Number of best checkpoints (by EMA SliceAP@50) to keep per trial. "
             "These are saved for later SWA weight averaging."
    )
    parser.add_argument(
        "--ckpt-min-ema", type=float, default=0.5,
        help="Minimum EMA SliceAP@50 a trial must reach before any checkpoint "
             "is saved. Trials below this threshold skip disk writes entirely.",
    )
    parser.add_argument(
        "--worker-jitter", type=float, default=0.0,
        help="Sleep this many seconds before starting.  Set automatically by "
             "the SLURM array task ID to stagger DB writes on study creation."
    )
    return parser.parse_args()


# ─── Augmentation builder ──────────────────────────────────────────────────────

def build_trial_transform(
    base_config: dict,
    use_clahe: bool,
    use_ssr: bool,
    use_rbc: bool,
    use_median_blur: bool,
    median_blur_limit: int,
    use_speckle_noise: bool,
    speckle_level: str,
):
    """Build an albumentations Compose pipeline for one trial.

    Always included (taken from base_config):
        HorizontalFlip, VerticalFlip

    Toggled by boolean trial flags:
        CLAHE                   (use_clahe)
        ShiftScaleRotate        (use_ssr)
        RandomBrightnessContrast (use_rbc)

    New in v3+, toggled by boolean trial flags:
        MedianBlur              (use_median_blur) — replaces GaussianBlur
        MultiplicativeNoise     (use_speckle_noise) — replaces GaussNoise;
                                intensity controlled by speckle_level

    Explicitly excluded regardless of base_config:
        GaussianBlur, GaussNoise
    """
    import albumentations as A

    tunable_flags = {
        "CLAHE": use_clahe,
        "ShiftScaleRotate": use_ssr,
        "RandomBrightnessContrast": use_rbc,
    }

    aug_list = []
    for aug in base_config.get("augmentations", []):
        for name, params in aug.items():
            if name in _REMOVED_AUGS:
                continue
            if name in tunable_flags:
                if tunable_flags[name]:
                    aug_list.append(getattr(A, name)(**params))
            else:
                aug_list.append(getattr(A, name)(**params))

    if use_median_blur:
        aug_list.append(A.MedianBlur(blur_limit=median_blur_limit, p=0.3))

    if use_speckle_noise:
        lo, hi = _SPECKLE_LEVELS[speckle_level]
        aug_list.append(A.MultiplicativeNoise(multiplier=(lo, hi), p=0.4))

    if not aug_list:
        return None
    return A.Compose(
        aug_list,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )


# ─── Checkpoint helpers ────────────────────────────────────────────────────────

class TopKCheckpointManager:
    """Maintains the top-K model checkpoints by EMA SliceAP@50 for one trial.

    Internally uses a min-heap so that when the buffer exceeds K entries the
    checkpoint with the lowest EMA SliceAP@50 is evicted and its file deleted.
    At the end of the trial call write_manifest() to persist a YAML index of
    the surviving checkpoints for later SWA weight averaging.

    Parameters
    ----------
    trial_dir : str
        Directory where .pt checkpoint files are written.
    k : int
        Maximum number of checkpoints to keep.
    trial_number : int
        Optuna trial number (stored in checkpoint metadata).
    trial_params : dict
        Full dict of trial hyperparameters (stored in checkpoint metadata).
    """

    def __init__(self, trial_dir: str, k: int, trial_number: int, trial_params: dict):
        self.trial_dir = trial_dir
        self.k = k
        self.trial_number = trial_number
        self.trial_params = trial_params
        # min-heap of (ema_slice_ap50, epoch, path, raw_slice_ap50)
        self._heap: list = []
        os.makedirs(trial_dir, exist_ok=True)

    def update(
        self,
        model: torch.nn.Module,
        epoch: int,
        slice_ap50: float,
        ema_slice_ap50: float,
    ) -> None:
        """Save a checkpoint and evict the worst one if the buffer is full.

        Only the base YOLOE model weights (model.model.state_dict()) are saved,
        not the loss wrapper — this keeps files smaller and makes them directly
        loadable by build_yoloe() for inference or SWA averaging.
        """
        fname = f"epoch{epoch:03d}_sliceap{ema_slice_ap50:.4f}.pt"
        path = os.path.join(self.trial_dir, fname)
        tmp_path = path + ".tmp"
        try:
            torch.save(
                {
                    "model_state_dict": model.model.state_dict(),
                    "epoch": epoch,
                    "slice_ap50": slice_ap50,
                    "ema_slice_ap50": ema_slice_ap50,
                    "trial_number": self.trial_number,
                    "trial_params": self.trial_params,
                },
                tmp_path,
            )
            os.replace(tmp_path, path)  # atomic rename — no partial .pt files
        except Exception as exc:
            print(f"  [CheckpointManager] WARNING: failed to save epoch {epoch} "
                  f"checkpoint ({exc}); skipping — trial continues.")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return

        heapq.heappush(self._heap, (ema_slice_ap50, epoch, path, slice_ap50))

        if len(self._heap) > self.k:
            _, _, evicted_path, _ = heapq.heappop(self._heap)  # removes worst
            if os.path.exists(evicted_path):
                os.remove(evicted_path)

    def write_manifest(self) -> str:
        """Write swa_manifest.yaml listing the surviving top-K checkpoints.

        Returns the path to the written manifest file.  The manifest is
        sorted by EMA SliceAP@50 descending so the best checkpoint is first.
        """
        entries = sorted(
            self._heap, key=lambda x: x[0], reverse=True
        )
        manifest = {
            "trial_number": self.trial_number,
            "trial_params": self.trial_params,
            "checkpoints": [
                {
                    "path": path,
                    "epoch": epoch,
                    "slice_ap50": round(float(raw_ap), 6),
                    "ema_slice_ap50": round(float(ema_ap), 6),
                }
                for ema_ap, epoch, path, raw_ap in entries
            ],
        }
        manifest_path = os.path.join(self.trial_dir, "swa_manifest.yaml")
        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
        return manifest_path

    def cleanup(self) -> None:
        """Delete all checkpoint files (called when a trial is pruned)."""
        for _, _, path, _ in self._heap:
            if os.path.exists(path):
                os.remove(path)
        self._heap.clear()


# ─── Trial runner ──────────────────────────────────────────────────────────────

def run_trial(
    trial: optuna.Trial,
    base_config: dict,
    epochs: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    ema_alpha: float,
    base_seed: int,
    checkpoint_dir: str,
    ckpt_top_k: int,
    ckpt_min_ema: float,
) -> float:
    """Run one Optuna trial.

    Objective
    ---------
    Returns the best EMA SliceAP@50 reached during training.  SliceAP@50 is
    used instead of box-level AP@50 because it has 357 samples (one per val
    slice) rather than a smaller number of GT boxes, making it more stable
    across epochs.  The EMA further smooths single-epoch noise without
    introducing the lag that would confuse the pruner if reported directly.

    Pruner interaction
    ------------------
    Raw (non-EMA) SliceAP@50 is reported to Optuna at each epoch so that the
    HyperbandPruner's rung comparisons reflect the true model state rather than
    a lagged EMA value.

    Early stopping
    --------------
    Triggered when EMA SliceAP@50 has not improved by more than
    early_stopping_min_delta for early_stopping_patience consecutive epochs.
    Using the EMA here prevents early termination on transient bad checkpoints
    (the ±0.1 swings visible in v3 logs).

    Checkpoints
    -----------
    The top-K checkpoints by EMA SliceAP@50 are saved to
    <checkpoint_dir>/trial_<N>/  during training.  If the trial is pruned all
    checkpoints are deleted to keep disk usage bounded.  On completion a
    swa_manifest.yaml is written listing the surviving checkpoint paths.
    """
    # ── Hyperparameter suggestions ────────────────────────────────────────────
    # Continuous
    lr           = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay",  1e-4, 1e-2, log=True)
    mosaic_prob  = trial.suggest_float("mosaic_prob",   0.1,  0.5)
    label_smooth = trial.suggest_float("label_smooth",  0.0,  0.15)
    copy_paste_p = trial.suggest_float("copy_paste_p",  0.0,  0.7)

    # Categorical
    slice_stride = trial.suggest_categorical("slice_stride", [5, 7, 10])
    batch_size   = trial.suggest_categorical("batch_size",   [8, 16, 32])

    # Augmentation toggles (inherited from v3)
    use_clahe   = trial.suggest_categorical("use_clahe",                [True, False])
    use_ssr     = trial.suggest_categorical("use_shift_scale_rotate",   [True, False])
    use_rbc     = trial.suggest_categorical("use_brightness_contrast",  [True, False])

    # New augmentation parameters (v3+)
    use_median_blur  = trial.suggest_categorical("use_median_blur",  [True, False])
    median_blur_limit = trial.suggest_categorical("median_blur_limit", [3, 5, 7])
    use_speckle_noise = trial.suggest_categorical("use_speckle_noise", [True, False])
    speckle_level     = trial.suggest_categorical("speckle_level", ["light", "medium", "heavy"])

    # ── Build transforms ──────────────────────────────────────────────────────
    train_transform = build_trial_transform(
        base_config, use_clahe, use_ssr, use_rbc,
        use_median_blur, median_blur_limit,
        use_speckle_noise, speckle_level,
    )

    cp_cfg = base_config.get("copy_paste", {})
    copy_paste = (
        TissueAwareCopyPaste(
            p=copy_paste_p,
            n_candidates=cp_cfg.get("n_candidates", 50),
            std_tol=cp_cfg.get("std_tol", 25.0),
            radius=cp_cfg.get("radius", 30),
            min_box_area=cp_cfg.get("min_box_area", 256),
            max_pastes=cp_cfg.get("max_pastes", 2),
            tissue_threshold=cp_cfg.get("tissue_threshold", 10),
        )
        if copy_paste_p > 0.0
        else None
    )

    trial_seed = base_seed + trial.number
    set_seed(trial_seed)

    device      = torch.device(base_config.get("device", "cuda"))
    num_workers = base_config.get("num_workers", 8)
    data_dir    = base_config["data_dir"]
    model_size  = base_config.get("model_size", "s")
    num_classes = base_config.get("num_classes", 1)
    use_pu_loss = base_config.get("use_pu_loss", False)

    print(
        f"\n[Trial {trial.number}] "
        f"lr={lr:.2e}  wd={weight_decay:.2e}  stride={slice_stride}  "
        f"bs={batch_size}  mosaic={mosaic_prob:.2f}  ls={label_smooth:.3f}  "
        f"clahe={use_clahe}  ssr={use_ssr}  rbc={use_rbc}  "
        f"median={use_median_blur}(limit={median_blur_limit})  "
        f"speckle={use_speckle_noise}({speckle_level})  cp_p={copy_paste_p:.2f}"
    )

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_dataset = UltrasoundDataset(
        root_dir=data_dir, split="train",
        transform=train_transform, mosaic_prob=mosaic_prob,
        copy_paste=copy_paste,
    )
    val_dataset = UltrasoundDataset(root_dir=data_dir, split="val")

    train_sampler = (
        ScanAwareSampler(train_dataset, stride=slice_stride, seed=trial_seed)
        if slice_stride > 1 else None
    )

    g = torch.Generator()
    g.manual_seed(trial_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model, optimiser, scheduler ───────────────────────────────────────────
    base_model = build_yoloe(model_size=model_size, num_classes=num_classes)
    model = YOLOEWithLoss(
        model=base_model,
        num_classes=num_classes,
        use_pu_loss=use_pu_loss,
        label_smooth=label_smooth,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_epochs = base_config.get("warmup_epochs", 8)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-5
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # ── Checkpoint manager ────────────────────────────────────────────────────
    trial_ckpt_dir = os.path.join(checkpoint_dir, f"trial_{trial.number:04d}")
    ckpt_manager = TopKCheckpointManager(
        trial_dir=trial_ckpt_dir,
        k=ckpt_top_k,
        trial_number=trial.number,
        trial_params=dict(trial.params),
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_ema  = -1.0      # best EMA SliceAP@50 seen so far (trial objective)
    ema_value = None      # running EMA; None until first epoch completes
    es_counter = 0        # epochs since last EMA improvement

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, metrics = validate(model, val_loader, device)

        raw_slice_ap = metrics["slice_ap_50"]
        raw_ap50     = metrics["map_50"]

        # EMA update — warm start on the first epoch (no lag from zero)
        if ema_value is None:
            ema_value = raw_slice_ap
        else:
            ema_value = ema_alpha * raw_slice_ap + (1.0 - ema_alpha) * ema_value

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        print(
            f"  Epoch {epoch:3d}/{epochs} | LR={current_lr:.2e} | "
            f"TLoss={train_loss:.4f} | VLoss={val_loss:.4f} | "
            f"AP@50={raw_ap50:.4f} | SliceAP={raw_slice_ap:.4f} | "
            f"EMA_SliceAP={ema_value:.4f}"
        )

        # Report RAW SliceAP to the pruner so rung comparisons are not lagged
        trial.report(raw_slice_ap, epoch)
        if trial.should_prune():
            print(f"  [Pruned at epoch {epoch}]")
            ckpt_manager.cleanup()
            raise optuna.exceptions.TrialPruned()

        # Save checkpoint — manager keeps only the top-K by EMA SliceAP
        if ema_value >= ckpt_min_ema:
            ckpt_manager.update(model, epoch, raw_slice_ap, ema_value)

        # Early stopping on EMA to avoid reacting to single-epoch dips
        if ema_value > best_ema + early_stopping_min_delta:
            best_ema = ema_value
            es_counter = 0
        else:
            es_counter += 1
            if early_stopping_patience > 0 and es_counter >= early_stopping_patience:
                print(
                    f"  [Early stop at epoch {epoch}, "
                    f"best EMA SliceAP={best_ema:.4f}]"
                )
                break

    # Write checkpoint manifest for SWA use
    manifest_path = ckpt_manager.write_manifest()
    print(f"  [Checkpoints] top-{ckpt_top_k} saved → {manifest_path}")

    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()

    # Return best EMA SliceAP@50 as the Optuna trial value
    return best_ema


# ─── Output builders (YAML) ────────────────────────────────────────────────────

def _build_output_augmentations(base_config: dict, params: dict) -> list:
    """Reconstruct the augmentation list for a completed trial's YAML config.

    GaussianBlur and GaussNoise are never included.
    Boolean-flag augmentations are included only if their flag is True.
    MedianBlur and MultiplicativeNoise are appended if their toggles are True.
    """
    tunable_flags = {
        "CLAHE":                    params.get("use_clahe", True),
        "ShiftScaleRotate":         params.get("use_shift_scale_rotate", True),
        "RandomBrightnessContrast": params.get("use_brightness_contrast", True),
    }

    aug_list = []
    for aug in base_config.get("augmentations", []):
        for name, aug_params in aug.items():
            if name in _REMOVED_AUGS:
                continue
            if name in tunable_flags:
                if tunable_flags[name]:
                    aug_list.append({name: aug_params})
            else:
                aug_list.append({name: aug_params})

    if params.get("use_median_blur", False):
        aug_list.append({"MedianBlur": {
            "blur_limit": params.get("median_blur_limit", 5), "p": 0.3
        }})

    if params.get("use_speckle_noise", False):
        lo, hi = _SPECKLE_LEVELS[params.get("speckle_level", "medium")]
        aug_list.append({"MultiplicativeNoise": {"multiplier": [lo, hi], "p": 0.4}})

    return aug_list


def _build_output_copy_paste(base_config: dict, params: dict) -> dict:
    """Reconstruct the copy_paste config block for a completed trial."""
    cp_p   = params.get("copy_paste_p", 0.0)
    cp_cfg = base_config.get("copy_paste", {})
    return {
        "enabled":    cp_p > 0.0,
        "p":          round(float(cp_p), 4),
        "std_tol":    cp_cfg.get("std_tol", 25.0),
        "max_pastes": cp_cfg.get("max_pastes", 2),
    }


# ─── Config saving ─────────────────────────────────────────────────────────────

def save_configs(study: optuna.Study, base_config: dict, output_dir: str, save_top_k: int):
    """Save top-K completed trials as YAML configs plus best_hyperparams_v4.yaml.

    The trial objective is EMA SliceAP@50, so ranked configs and the best
    config are ranked and annotated by that metric (key: _optuna_slice_ap50).
    """
    os.makedirs(output_dir, exist_ok=True)

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        print("No completed trials to save.")
        return

    completed.sort(key=lambda t: t.value, reverse=True)
    top_k = completed[:save_top_k]

    print(f"\nSaving top-{len(top_k)} configs to {output_dir}/")
    for rank, trial in enumerate(top_k, start=1):
        config = dict(base_config)
        config.update(trial.params)
        for flag in _TRIAL_FLAGS:
            config.pop(flag, None)
        config["augmentations"] = _build_output_augmentations(base_config, trial.params)
        config["copy_paste"]    = _build_output_copy_paste(base_config, trial.params)
        config["_optuna_trial_number"] = trial.number
        config["_optuna_slice_ap50"]   = round(float(trial.value), 6)

        fname = f"hparam_v4_rank{rank:02d}_sliceap{trial.value:.4f}.yaml"
        path  = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"  Rank {rank:2d} | SliceAP={trial.value:.4f} | {trial.params} → {fname}")

    # Fixed-name best config for direct use with train.py --config
    best = top_k[0]
    best_config = dict(base_config)
    best_config.update(best.params)
    for flag in _TRIAL_FLAGS:
        best_config.pop(flag, None)
    best_config["augmentations"] = _build_output_augmentations(base_config, best.params)
    best_config["copy_paste"]    = _build_output_copy_paste(base_config, best.params)
    best_config.pop("_optuna_trial_number", None)
    best_config.pop("_optuna_slice_ap50",   None)

    best_path = os.path.join(output_dir, "best_hyperparams_v4.yaml")
    with open(best_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
    print(f"\nBest config (EMA SliceAP@50={best.value:.4f}) → {best_path}")


# ─── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.worker_jitter > 0:
        time.sleep(args.worker_jitter)

    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    if args.storage is None:
        studies_dir = os.path.join(_project_root, "optuna_studies")
        os.makedirs(studies_dir, exist_ok=True)
        args.storage = f"sqlite:///{studies_dir}/{args.study_name}.db"

    # Enable WAL mode once per process before Optuna touches the DB.
    # WAL lets readers proceed while one writer holds the lock, which cuts
    # contention among the 20 parallel SLURM workers significantly.
    if args.storage.startswith("sqlite:///"):
        import sqlite3 as _sqlite3
        _db_path = args.storage[len("sqlite:///"):]
        with _sqlite3.connect(_db_path, timeout=60) as _conn:
            _conn.execute("PRAGMA journal_mode=WAL")

    print(f"Study name   : {args.study_name}")
    print(f"Storage      : {args.storage}")
    print(f"Trials       : {args.n_trials} (this worker)")
    print(f"Epochs/trial : {args.epochs} (max, EMA early stopping patience={args.early_stopping_patience})")
    print(f"EMA alpha    : {args.ema_alpha}")
    print(f"Checkpoints  : top-{args.ckpt_top_k} per trial → {args.checkpoint_dir}")

    sampler = TPESampler(seed=args.seed)

    # HyperbandPruner: rungs at epochs 10, 30, 50 (reduction_factor=3).
    # Keeps top 1/3 of trials at each rung.  min_resource=10 ensures no trial
    # is pruned before epoch 10 (past the warmup phase where all APs are near 0).
    pruner = HyperbandPruner(
        min_resource=10,
        max_resource=args.epochs,
        reduction_factor=3,
    )

    # For SQLite on NFS with many parallel workers, the default lock timeout is
    # 0 ms — causing immediate "disk I/O error" on contention.  RDBStorage with
    # connect_args timeout=25 makes SQLite queue waiting writers for up to 25 s
    # instead of failing instantly.
    if args.storage.startswith("sqlite:///"):
        storage = RDBStorage(
            url=args.storage,
            engine_kwargs={
                "connect_args": {"timeout": 25},
            },
        )
    else:
        storage = args.storage

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: run_trial(
            trial,
            base_config=base_config,
            epochs=args.epochs,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            ema_alpha=args.ema_alpha,
            base_seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            ckpt_top_k=args.ckpt_top_k,
            ckpt_min_ema=args.ckpt_min_ema,
        ),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\n=== Worker done | Completed: {len(completed)} | Pruned: {len(pruned)} ===")
    if study.best_trial:
        print(f"Best EMA SliceAP@50 across all workers: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

    save_configs(study, base_config, args.output_config_dir, args.save_top_k)


if __name__ == "__main__":
    main()
