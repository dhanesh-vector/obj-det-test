"""
YOLOE Hyperparameter Tuning v5 with Optuna.

═══════════════════════════════════════════════════════════════════════════════
Changes from v4
═══════════════════════════════════════════════════════════════════════════════

Motivation
──────────
v4 ran 300 trials (job 752) and achieved best EMA SliceAP@50 = 0.7280.
Post-hoc analysis of those logs revealed three consistent overfitting signals:

  1. The train/val loss gap triples from ~9 at epoch 5 to ~26 by the final
     epoch across virtually all trials, indicating the model memorises
     scan-specific features of the 13 training patients.

  2. Val loss bottoms at epoch 16–30 in nearly every trial and drifts up
     by +1–7 units afterward. Running 50 epochs wastes compute and deepens
     the gap without improving SliceAP@50.

  3. The best weight_decay in both v3 and v4 (7.97e-03) saturated against
     the search space upper bound of 1e-2 — a clear signal that higher
     regularisation is beneficial but was previously unconstrained.

Search space changes
────────────────────
  weight_decay   [1e-4, 1e-2 log] → [1e-4, 5e-2 log]
    Rationale: Both v3 and v4 best trials found wd=7.97e-03, right at the
    old ceiling.  Extending the boundary lets TPE explore stronger
    regularisation (up to 5×) without biasing toward the old range.

  mosaic_prob    [0.1, 0.5]       → [0.3, 0.8]
    Rationale: Cross-scan mosaic is the strongest anti-overfitting
    augmentation available — each quadrant comes from a different patient,
    forcing the model to generalise across scan-specific textures.  The v4
    best found 0.39, which is near the old upper boundary.  Extending to
    0.8 allows more aggressive use. At high rates, lesion boxes appear at
    half scale (×0.5), which also acts as implicit scale regularisation.

  copy_paste_p   [0.0, 0.7]       → [0.2, 0.7]
    Rationale: v4 best used copy_paste_p=0.11 — well below where it would
    be most effective.  With only 13 training scans, copy-paste is the
    only source of novel lesion instances per epoch.  Setting a floor of
    0.2 ensures every trial uses it; the ceiling is unchanged.

Training budget changes
───────────────────────
  --epochs        50  → 35
    Rationale: Val loss minimum is reached at epoch 16–30 in virtually all
    v4 trials. Epochs 35–50 were exclusively deepening the overfitting gap
    with no SliceAP improvement.  Reducing max epochs saves ~30% of
    per-trial GPU time, enabling more trials in the same wall time.

  --early-stopping-patience   15  → 8
    Rationale: With EMA smoothing (α=0.3, half-life ~2 epochs) a 15-epoch
    patience window is equivalent to ~30 raw epochs of tolerance — far too
    permissive for a val-loss-floored regime.  8 epochs provides a
    tighter, faster exit while still allowing the EMA to recover from a
    single bad checkpoint.

Unchanged from v4
─────────────────
  Objective:     EMA SliceAP@50 (α=0.3)
  Pruner:        HyperbandPruner (min=10, max=35, reduction_factor=3)
                 Note: max_resource updated to match new --epochs=35
  Sampler:       TPESampler
  Checkpoints:   Top-K per trial via TopKCheckpointManager
  All other search space parameters: unchanged from v4

═══════════════════════════════════════════════════════════════════════════════
Search space (v5)
═══════════════════════════════════════════════════════════════════════════════
Continuous:   learning_rate  [1e-4, 5e-3 log]    (unchanged)
              weight_decay   [1e-4, 5e-2 log]     ← extended upper bound
              mosaic_prob    [0.3, 0.8]            ← raised floor and ceiling
              label_smooth   [0.0, 0.15]           (unchanged)
              copy_paste_p   [0.2, 0.7]            ← raised floor
Categorical:  slice_stride   {5, 7, 10}            (unchanged)
              batch_size     {8, 16, 32}            (unchanged)
              use_clahe              {T/F}           (unchanged)
              use_shift_scale_rotate {T/F}           (unchanged)
              use_brightness_contrast {T/F}          (unchanged)
              use_median_blur        {T/F}           (unchanged)
              median_blur_limit      {3, 5, 7}       (unchanged)
              use_speckle_noise      {T/F}           (unchanged)
              speckle_level          {light,medium,heavy} (unchanged)
Fixed:        HorizontalFlip, VerticalFlip always on.
Removed:      GaussianBlur, GaussNoise (removed in v3).

═══════════════════════════════════════════════════════════════════════════════
Usage
═══════════════════════════════════════════════════════════════════════════════
Single node:
    python train/tune_hyperparams_v5.py --config train/config/default.yaml \\
        --n-trials 40

Multi-node via SLURM array:
    sbatch slurm-script/tune_hyperparams_v5.sh
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

_REMOVED_AUGS = {"GaussianBlur", "GaussNoise"}

_SPECKLE_LEVELS = {
    "light":  [0.95, 1.05],
    "medium": [0.90, 1.10],
    "heavy":  [0.80, 1.20],
}

_TRIAL_FLAGS = {
    "use_clahe", "use_shift_scale_rotate", "use_brightness_contrast",
    "use_median_blur", "median_blur_limit",
    "use_speckle_noise", "speckle_level",
    "copy_paste_p",
}


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOE Hyperparameter Tuning v5 — extended weight_decay & "
                    "mosaic search space, copy-paste floor, reduced epoch budget"
    )
    parser.add_argument(
        "--config", type=str, default="train/config/default.yaml",
    )
    parser.add_argument(
        "--study-name", type=str, default="yoloe_hparam_search_v5",
    )
    parser.add_argument(
        "--storage", type=str, default=None,
    )
    parser.add_argument(
        "--n-trials", type=int, default=5,
    )
    parser.add_argument(
        "--epochs", type=int, default=35,
        help="Max epochs per trial. Reduced from 50 (v4): val loss bottoms at "
             "ep 16–30 in virtually all v4 trials; ep 35–50 only deepens "
             "the train/val gap."
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=8,
        help="Epochs without EMA SliceAP@50 improvement before a trial is "
             "stopped early. Tightened from 15 (v4): EMA smoothing already "
             "absorbs single-epoch dips; 8 epochs is sufficient tolerance."
    )
    parser.add_argument(
        "--early-stopping-min-delta", type=float, default=0.005,
    )
    parser.add_argument(
        "--ema-alpha", type=float, default=0.3,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--save-top-k", type=int, default=5,
    )
    parser.add_argument(
        "--output-config-dir", type=str, default="train/config",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints/hparam_v5",
    )
    parser.add_argument(
        "--ckpt-top-k", type=int, default=3,
    )
    parser.add_argument(
        "--ckpt-min-ema", type=float, default=0.5,
        help="Minimum EMA SliceAP@50 a trial must reach before any checkpoint "
             "is saved. Trials below this threshold skip disk writes entirely.",
    )
    parser.add_argument(
        "--worker-jitter", type=float, default=0.0,
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
    """Maintains the top-K model checkpoints by EMA SliceAP@50 for one trial."""

    def __init__(self, trial_dir: str, k: int, trial_number: int, trial_params: dict):
        self.trial_dir = trial_dir
        self.k = k
        self.trial_number = trial_number
        self.trial_params = trial_params
        self._heap: list = []
        os.makedirs(trial_dir, exist_ok=True)

    def update(self, model, epoch: int, slice_ap50: float, ema_slice_ap50: float):
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
            _, _, evicted_path, _ = heapq.heappop(self._heap)
            if os.path.exists(evicted_path):
                os.remove(evicted_path)

    def write_manifest(self) -> str:
        entries = sorted(self._heap, key=lambda x: x[0], reverse=True)
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

    def cleanup(self):
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
    # ── Hyperparameter suggestions ────────────────────────────────────────────
    lr           = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    # v5: upper bound extended from 1e-2 → 5e-2 (v4 best saturated at 7.97e-03)
    weight_decay = trial.suggest_float("weight_decay",  1e-4, 5e-2, log=True)
    # v5: floor raised from 0.1 → 0.3, ceiling raised from 0.5 → 0.8
    mosaic_prob  = trial.suggest_float("mosaic_prob",   0.3,  0.8)
    label_smooth = trial.suggest_float("label_smooth",  0.0,  0.15)
    # v5: floor raised from 0.0 → 0.2 (forces copy-paste in every trial)
    copy_paste_p = trial.suggest_float("copy_paste_p",  0.2,  0.7)

    slice_stride = trial.suggest_categorical("slice_stride", [5, 7, 10])
    batch_size   = trial.suggest_categorical("batch_size",   [8, 16, 32])

    use_clahe    = trial.suggest_categorical("use_clahe",               [True, False])
    use_ssr      = trial.suggest_categorical("use_shift_scale_rotate",  [True, False])
    use_rbc      = trial.suggest_categorical("use_brightness_contrast", [True, False])

    use_median_blur   = trial.suggest_categorical("use_median_blur",  [True, False])
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
    copy_paste = TissueAwareCopyPaste(
        p=copy_paste_p,
        n_candidates=cp_cfg.get("n_candidates", 50),
        std_tol=cp_cfg.get("std_tol", 25.0),
        radius=cp_cfg.get("radius", 30),
        min_box_area=cp_cfg.get("min_box_area", 256),
        max_pastes=cp_cfg.get("max_pastes", 2),
        tissue_threshold=cp_cfg.get("tissue_threshold", 10),
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
    best_ema  = -1.0
    ema_value = None
    es_counter = 0

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, metrics = validate(model, val_loader, device)

        raw_slice_ap = metrics["slice_ap_50"]
        raw_ap50     = metrics["map_50"]

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

        trial.report(raw_slice_ap, epoch)
        if trial.should_prune():
            print(f"  [Pruned at epoch {epoch}]")
            ckpt_manager.cleanup()
            raise optuna.exceptions.TrialPruned()

        if ema_value >= ckpt_min_ema:
            ckpt_manager.update(model, epoch, raw_slice_ap, ema_value)

        if ema_value > best_ema + early_stopping_min_delta:
            best_ema = ema_value
            es_counter = 0
        else:
            es_counter += 1
            if early_stopping_patience > 0 and es_counter >= early_stopping_patience:
                print(f"  [Early stop at epoch {epoch}, best EMA SliceAP={best_ema:.4f}]")
                break

    manifest_path = ckpt_manager.write_manifest()
    print(f"  [Checkpoints] top-{ckpt_top_k} saved → {manifest_path}")

    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()

    return best_ema


# ─── Output builders ───────────────────────────────────────────────────────────

def _build_output_augmentations(base_config: dict, params: dict) -> list:
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
    cp_p   = params.get("copy_paste_p", 0.2)
    cp_cfg = base_config.get("copy_paste", {})
    return {
        "enabled":    True,   # always enabled in v5 (floor > 0)
        "p":          round(float(cp_p), 4),
        "std_tol":    cp_cfg.get("std_tol", 25.0),
        "max_pastes": cp_cfg.get("max_pastes", 2),
    }


# ─── Config saving ─────────────────────────────────────────────────────────────

def save_configs(study: optuna.Study, base_config: dict, output_dir: str, save_top_k: int):
    os.makedirs(output_dir, exist_ok=True)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
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

        fname = f"hparam_v5_rank{rank:02d}_sliceap{trial.value:.4f}.yaml"
        path  = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"  Rank {rank:2d} | SliceAP={trial.value:.4f} | {trial.params} → {fname}")

    best = top_k[0]
    best_config = dict(base_config)
    best_config.update(best.params)
    for flag in _TRIAL_FLAGS:
        best_config.pop(flag, None)
    best_config["augmentations"] = _build_output_augmentations(base_config, best.params)
    best_config["copy_paste"]    = _build_output_copy_paste(base_config, best.params)
    best_config.pop("_optuna_trial_number", None)
    best_config.pop("_optuna_slice_ap50",   None)

    best_path = os.path.join(output_dir, "best_hyperparams_v5.yaml")
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
        with _sqlite3.connect(_db_path, timeout=25) as _conn:
            _conn.execute("PRAGMA journal_mode=WAL")

    print(f"Study name   : {args.study_name}")
    print(f"Storage      : {args.storage}")
    print(f"Trials       : {args.n_trials} (this worker)")
    print(f"Epochs/trial : {args.epochs} (max, EMA early stopping patience={args.early_stopping_patience})")
    print(f"EMA alpha    : {args.ema_alpha}")
    print(f"Checkpoints  : top-{args.ckpt_top_k} per trial → {args.checkpoint_dir}")

    sampler = TPESampler(seed=args.seed)

    # HyperbandPruner: max_resource updated to match new --epochs=35.
    # Rungs at epochs 10 and 35 (reduction_factor=3 → keeps top ⅓ at each rung).
    pruner = HyperbandPruner(
        min_resource=10,
        max_resource=args.epochs,
        reduction_factor=3,
    )

    # For SQLite on NFS with many parallel workers, the default lock timeout is
    # 0 ms — causing immediate "disk I/O error" on contention.  RDBStorage with
    # connect_args timeout=60 makes SQLite queue waiting writers for up to 60 s
    # instead of failing instantly.  WAL mode further improves throughput by
    # letting readers proceed concurrently with the single active writer.
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
