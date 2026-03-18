"""
YOLOE Hyperparameter Tuning v3 with Optuna.

Changes from v2:
- Removed GaussianBlur and GaussNoise from augmentation search space.
- Added MedianBlur with tunable blur_limit (3, 5, 7) and on/off toggle.
- Added MultiplicativeNoise (speckle) with tunable intensity level
  ("light"=[0.95,1.05], "medium"=[0.9,1.1], "heavy"=[0.8,1.2]) and on/off toggle.
- Added TissueAwareCopyPaste with tunable probability (0.0–0.7); 0.0 disables it.

Optimises: learning_rate, weight_decay, slice_stride, batch_size, mosaic_prob,
           label_smooth, and per-augmentation on/off flags for CLAHE,
           ShiftScaleRotate, RandomBrightnessContrast, MedianBlur,
           MultiplicativeNoise (speckle), and copy-paste probability.

Designed for distributed search: multiple SLURM workers share one SQLite study on NFS.
Each worker runs --n-trials trials independently and contributes to the same study.
After all trials, each worker saves the top-K configs as YAML to train/config/,
and the single best is saved as best_hyperparams_v3.yaml.

Usage (single node):
    python train/tune_hyperparams_v3.py --config train/config/default.yaml --n-trials 40

Usage (multi-node via SLURM array):
    sbatch slurm-script/tune_hyperparams_v3.sh
"""

import sys
import os
import argparse
import time
import random
import yaml

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
# Insert train/ so `import train` resolves to train/train.py (not the directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.yoloe import build_yoloe, YOLOEWithLoss
from data.dataset import UltrasoundDataset, collate_fn
from data.sampler import ScanAwareSampler
from data.augmentations import TissueAwareCopyPaste
from train import train_one_epoch, validate, set_seed, seed_worker

# Augmentations removed from the search space in v3 (replaced by MedianBlur + speckle)
_REMOVED_AUGS = {"GaussianBlur", "GaussNoise"}

# Mapping from speckle_level trial value → MultiplicativeNoise multiplier range
_SPECKLE_LEVELS = {
    "light":  [0.95, 1.05],
    "medium": [0.90, 1.10],
    "heavy":  [0.80, 1.20],
}


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOE Hyperparameter Tuning v3 with Optuna")
    parser.add_argument(
        "--config", type=str, default="train/config/default.yaml",
        help="Base config YAML (data_dir, model_size, etc. are taken from here unchanged)"
    )
    parser.add_argument(
        "--study-name", type=str, default="yoloe_hparam_search_v3",
        help="Optuna study name — shared across all parallel workers"
    )
    parser.add_argument(
        "--storage", type=str, default=None,
        help="Optuna storage URL. Defaults to sqlite:///optuna_studies/<study_name>.db"
    )
    parser.add_argument(
        "--n-trials", type=int, default=5,
        help="Number of trials this worker will run"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Max epochs per trial (early stopping will usually cut this short)"
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=15,
        help="Epochs without AP@50 improvement before a trial is stopped early"
    )
    parser.add_argument(
        "--early-stopping-min-delta", type=float, default=0.005,
        help="Minimum AP@50 improvement to reset the early-stopping counter"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (each trial gets seed + trial.number for independence)"
    )
    parser.add_argument(
        "--save-top-k", type=int, default=5,
        help="Number of top configs to write as YAML after tuning completes"
    )
    parser.add_argument(
        "--output-config-dir", type=str, default="train/config",
        help="Directory where best YAML configs are written"
    )
    parser.add_argument(
        "--worker-jitter", type=float, default=0.0,
        help="Sleep this many seconds before starting (set from SLURM to stagger DB writes)"
    )
    return parser.parse_args()


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
    """Build albumentations Compose for a single trial.

    GaussianBlur and GaussNoise are removed entirely in v3.
    HorizontalFlip and VerticalFlip are always included.
    CLAHE, ShiftScaleRotate, RandomBrightnessContrast are toggled by trial flags.
    MedianBlur and MultiplicativeNoise (speckle) are new tunable augmentations.
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
            # Drop v2-era noise/blur — replaced by MedianBlur + speckle below
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


def run_trial(
    trial: optuna.Trial,
    base_config: dict,
    epochs: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    base_seed: int,
) -> float:
    """Run one training trial. Returns best val AP@50 achieved."""
    # --- Suggest hyperparameters (unchanged from v2) ---
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    slice_stride = trial.suggest_categorical("slice_stride", [5, 7, 10])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    mosaic_prob = trial.suggest_float("mosaic_prob", 0.1, 0.5)
    label_smooth = trial.suggest_float("label_smooth", 0.0, 0.15)
    use_clahe = trial.suggest_categorical("use_clahe", [True, False])
    use_ssr = trial.suggest_categorical("use_shift_scale_rotate", [True, False])
    use_rbc = trial.suggest_categorical("use_brightness_contrast", [True, False])

    # --- New v3 augmentation parameters ---
    use_median_blur = trial.suggest_categorical("use_median_blur", [True, False])
    median_blur_limit = trial.suggest_categorical("median_blur_limit", [3, 5, 7])
    use_speckle_noise = trial.suggest_categorical("use_speckle_noise", [True, False])
    speckle_level = trial.suggest_categorical("speckle_level", ["light", "medium", "heavy"])
    # copy_paste_p=0.0 disables tissue-aware copy-paste
    copy_paste_p = trial.suggest_float("copy_paste_p", 0.0, 0.7)

    train_transform = build_trial_transform(
        base_config, use_clahe, use_ssr, use_rbc,
        use_median_blur, median_blur_limit,
        use_speckle_noise, speckle_level,
    )

    # Build tissue-aware copy-paste (uses base_config params for everything except p)
    cp_cfg = base_config.get("copy_paste", {})
    if copy_paste_p > 0.0:
        copy_paste = TissueAwareCopyPaste(
            p=copy_paste_p,
            n_candidates=cp_cfg.get("n_candidates", 50),
            std_tol=cp_cfg.get("std_tol", 25.0),
            radius=cp_cfg.get("radius", 30),
            min_box_area=cp_cfg.get("min_box_area", 256),
            max_pastes=cp_cfg.get("max_pastes", 2),
            tissue_threshold=cp_cfg.get("tissue_threshold", 10),
        )
    else:
        copy_paste = None

    trial_seed = base_seed + trial.number
    set_seed(trial_seed)

    device = torch.device(base_config.get("device", "cuda"))
    num_workers = base_config.get("num_workers", 8)
    data_dir = base_config["data_dir"]
    model_size = base_config.get("model_size", "s")
    num_classes = base_config.get("num_classes", 1)
    use_pu_loss = base_config.get("use_pu_loss", False)

    print(
        f"\n[Trial {trial.number}] lr={lr:.2e}  wd={weight_decay:.2e}  "
        f"stride={slice_stride}  bs={batch_size}  mosaic={mosaic_prob:.2f}  "
        f"ls={label_smooth:.3f}  clahe={use_clahe}  ssr={use_ssr}  rbc={use_rbc}  "
        f"median={use_median_blur}(limit={median_blur_limit})  "
        f"speckle={use_speckle_noise}({speckle_level})  cp_p={copy_paste_p:.2f}"
    )

    # --- Datasets & loaders ---
    train_dataset = UltrasoundDataset(
        root_dir=data_dir, split="train",
        transform=train_transform, mosaic_prob=mosaic_prob,
        copy_paste=copy_paste,
    )
    val_dataset = UltrasoundDataset(root_dir=data_dir, split="val")

    if slice_stride > 1:
        train_sampler = ScanAwareSampler(train_dataset, stride=slice_stride, seed=trial_seed)
    else:
        train_sampler = None

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

    # --- Model ---
    base_model = build_yoloe(model_size=model_size, num_classes=num_classes)
    model = YOLOEWithLoss(
        model=base_model,
        num_classes=num_classes,
        use_pu_loss=use_pu_loss,
        label_smooth=label_smooth,
    ).to(device)

    # --- Optimizer + scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_epochs = base_config.get("warmup_epochs", 8)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-5
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    # --- Training loop ---
    best_ap50 = -1.0
    early_stopping_counter = 0

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, metrics = validate(model, val_loader, device)
        ap50 = metrics["map_50"]

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        print(
            f"  Epoch {epoch:3d}/{epochs} | LR={current_lr:.2e} | "
            f"TLoss={train_loss:.4f} | VLoss={val_loss:.4f} | AP@50={ap50:.4f}"
        )

        trial.report(ap50, epoch)
        if trial.should_prune():
            print(f"  [Pruned at epoch {epoch}]")
            raise optuna.exceptions.TrialPruned()

        if ap50 > best_ap50 + early_stopping_min_delta:
            best_ap50 = ap50
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_patience > 0 and early_stopping_counter >= early_stopping_patience:
                print(f"  [Early stop at epoch {epoch}, best AP@50={best_ap50:.4f}]")
                break

    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()

    return best_ap50


def _build_output_augmentations(base_config: dict, params: dict) -> list:
    """Reconstruct the augmentation list for a completed trial's YAML output.

    - GaussianBlur and GaussNoise are never included (removed in v3).
    - Standard boolean-flag augmentations (CLAHE, SSR, RBC) are included if True.
    - MedianBlur is included if use_median_blur=True, with the trial's blur_limit.
    - MultiplicativeNoise is included if use_speckle_noise=True, with the
      trial's speckle_level mapped to a multiplier range.
    """
    tunable_flags = {
        "CLAHE": params.get("use_clahe", True),
        "ShiftScaleRotate": params.get("use_shift_scale_rotate", True),
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
        blur_limit = params.get("median_blur_limit", 5)
        aug_list.append({"MedianBlur": {"blur_limit": blur_limit, "p": 0.3}})

    if params.get("use_speckle_noise", False):
        level = params.get("speckle_level", "medium")
        lo, hi = _SPECKLE_LEVELS[level]
        aug_list.append({"MultiplicativeNoise": {"multiplier": [lo, hi], "p": 0.4}})

    return aug_list


def _build_output_copy_paste(base_config: dict, params: dict) -> dict:
    """Reconstruct the copy_paste config block for a completed trial."""
    cp_p = params.get("copy_paste_p", 0.0)
    cp_cfg = base_config.get("copy_paste", {})
    return {
        "enabled": cp_p > 0.0,
        "p": round(float(cp_p), 4),
        "std_tol": cp_cfg.get("std_tol", 25.0),
        "max_pastes": cp_cfg.get("max_pastes", 2),
    }


# Optuna trial params that should not be written directly into the output config
_TRIAL_FLAGS = {
    "use_clahe", "use_shift_scale_rotate", "use_brightness_contrast",
    "use_median_blur", "median_blur_limit",
    "use_speckle_noise", "speckle_level",
    "copy_paste_p",
}


def save_configs(study: optuna.Study, base_config: dict, output_dir: str, save_top_k: int):
    """Save top-K completed trials as YAML configs and best_hyperparams_v3.yaml."""
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
        config["copy_paste"] = _build_output_copy_paste(base_config, trial.params)
        config["_optuna_trial_number"] = trial.number
        config["_optuna_ap50"] = round(float(trial.value), 6)

        fname = f"hparam_v3_rank{rank:02d}_ap50_{trial.value:.4f}.yaml"
        path = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"  Rank {rank:2d} | AP@50={trial.value:.4f} | {trial.params} → {fname}")

    # Fixed-name best config
    best = top_k[0]
    best_config = dict(base_config)
    best_config.update(best.params)
    for flag in _TRIAL_FLAGS:
        best_config.pop(flag, None)
    best_config["augmentations"] = _build_output_augmentations(base_config, best.params)
    best_config["copy_paste"] = _build_output_copy_paste(base_config, best.params)
    best_config.pop("_optuna_trial_number", None)
    best_config.pop("_optuna_ap50", None)

    best_path = os.path.join(output_dir, "best_hyperparams_v3.yaml")
    with open(best_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
    print(f"\nBest config (AP@50={best.value:.4f}) → {best_path}")


def main():
    args = parse_args()

    if args.worker_jitter > 0:
        time.sleep(args.worker_jitter)

    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    project_dir = _project_root
    if args.storage is None:
        studies_dir = os.path.join(project_dir, "optuna_studies")
        os.makedirs(studies_dir, exist_ok=True)
        args.storage = f"sqlite:///{studies_dir}/{args.study_name}.db"

    print(f"Study name : {args.study_name}")
    print(f"Storage    : {args.storage}")
    print(f"Trials     : {args.n_trials} (this worker)")
    print(f"Epochs/trial: {args.epochs} (max, early stopping at patience={args.early_stopping_patience})")

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: run_trial(
            trial, base_config,
            args.epochs, args.early_stopping_patience,
            args.early_stopping_min_delta, args.seed,
        ),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\n=== Worker done | Completed: {len(completed)} | Pruned: {len(pruned)} ===")
    if study.best_trial:
        print(f"Best AP@50 across all workers: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

    save_configs(study, base_config, args.output_config_dir, args.save_top_k)


if __name__ == "__main__":
    main()
