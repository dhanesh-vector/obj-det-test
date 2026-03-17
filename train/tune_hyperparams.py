"""
YOLOE Hyperparameter Tuning with Optuna.

Optimises: learning_rate, weight_decay, slice_stride, batch_size, mosaic_prob,
           label_smooth, and per-augmentation on/off flags for CLAHE,
           ShiftScaleRotate, and RandomBrightnessContrast.
Remaining augmentations (HorizontalFlip, VerticalFlip, GaussianBlur, GaussNoise)
are kept fixed from the base config.

Designed for distributed search: multiple SLURM workers share one SQLite study on NFS.
Each worker runs --n-trials trials independently and contributes to the same study.
After all trials, each worker saves the top-K configs as YAML to train/config/.

Usage (single node):
    python train/tune_hyperparams.py --config train/config/default.yaml --n-trials 40

Usage (multi-node via SLURM array):
    sbatch slurm-script/tune_hyperparams.sh
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
from train import train_one_epoch, validate, set_seed, seed_worker


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOE Hyperparameter Tuning with Optuna")
    parser.add_argument(
        "--config", type=str, default="train/config/default.yaml",
        help="Base config YAML (data_dir, model_size, augmentations, etc. are taken from here unchanged)"
    )
    parser.add_argument(
        "--study-name", type=str, default="yoloe_hparam_search",
        help="Optuna study name — shared across all parallel workers"
    )
    parser.add_argument(
        "--storage", type=str, default=None,
        help="Optuna storage URL. Defaults to sqlite:///optuna_studies/<study_name>.db inside the project root"
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


def _filter_augmentations(base_config: dict, trial_params: dict) -> list:
    """
    Rebuild the augmentation list from base_config respecting the boolean toggle flags
    stored in trial_params.  Augmentations not controlled by a flag are always included.
    """
    tunable = {
        "CLAHE": trial_params.get("use_clahe", True),
        "ShiftScaleRotate": trial_params.get("use_shift_scale_rotate", True),
        "RandomBrightnessContrast": trial_params.get("use_brightness_contrast", True),
    }
    return [
        aug for aug in base_config.get("augmentations", [])
        for name in aug
        if name not in tunable or tunable[name]
    ]


def build_trial_transform(base_config: dict, use_clahe: bool, use_ssr: bool, use_rbc: bool):
    """
    Build albumentations transform for a single trial.

    HorizontalFlip, VerticalFlip, GaussianBlur, GaussNoise are always included
    (taken from the base config).  CLAHE, ShiftScaleRotate, and
    RandomBrightnessContrast are toggled by the trial flags.
    """
    import albumentations as A

    # Which augmentations are controlled by trial flags
    tunable = {
        "CLAHE": use_clahe,
        "ShiftScaleRotate": use_ssr,
        "RandomBrightnessContrast": use_rbc,
    }

    aug_list = []
    for aug in base_config.get("augmentations", []):
        for name, params in aug.items():
            if name in tunable:
                if tunable[name]:
                    aug_list.append(getattr(A, name)(**params))
            else:
                aug_list.append(getattr(A, name)(**params))

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
    """
    Run one training trial. Returns best val AP@50 achieved.
    Raises optuna.exceptions.TrialPruned if the trial is pruned early.
    """
    # --- Suggest hyperparameters ---
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    slice_stride = trial.suggest_categorical("slice_stride", [5, 7, 10])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    mosaic_prob = trial.suggest_float("mosaic_prob", 0.1, 0.5)
    label_smooth = trial.suggest_float("label_smooth", 0.0, 0.15)
    use_clahe = trial.suggest_categorical("use_clahe", [True, False])
    use_ssr = trial.suggest_categorical("use_shift_scale_rotate", [True, False])
    use_rbc = trial.suggest_categorical("use_brightness_contrast", [True, False])

    train_transform = build_trial_transform(base_config, use_clahe, use_ssr, use_rbc)

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
        f"ls={label_smooth:.3f}  clahe={use_clahe}  ssr={use_ssr}  rbc={use_rbc}"
    )

    # --- Datasets & loaders ---
    train_dataset = UltrasoundDataset(
        root_dir=data_dir, split="train",
        transform=train_transform, mosaic_prob=mosaic_prob,
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

    # --- Optimizer + scheduler (same as train.py) ---
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

        # Report to Optuna for pruning
        trial.report(ap50, epoch)
        if trial.should_prune():
            print(f"  [Pruned at epoch {epoch}]")
            raise optuna.exceptions.TrialPruned()

        # Early stopping
        if ap50 > best_ap50 + early_stopping_min_delta:
            best_ap50 = ap50
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_patience > 0 and early_stopping_counter >= early_stopping_patience:
                print(f"  [Early stop at epoch {epoch}, best AP@50={best_ap50:.4f}]")
                break

    # Explicitly free GPU memory before next trial
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()

    return best_ap50


def save_configs(study: optuna.Study, base_config: dict, output_dir: str, save_top_k: int):
    """Save top-K completed trials as YAML configs to output_dir."""
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

    _bool_flags = {"use_clahe", "use_shift_scale_rotate", "use_brightness_contrast"}

    print(f"\nSaving top-{len(top_k)} configs to {output_dir}/")
    for rank, trial in enumerate(top_k, start=1):
        config = dict(base_config)
        config.update(trial.params)
        # Rebuild augmentation list honouring the boolean toggle flags, then drop the flags
        config["augmentations"] = _filter_augmentations(base_config, trial.params)
        for flag in _bool_flags:
            config.pop(flag, None)
        config["_optuna_trial_number"] = trial.number
        config["_optuna_ap50"] = round(float(trial.value), 6)

        fname = f"hparam_rank{rank:02d}_ap50_{trial.value:.4f}.yaml"
        path = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"  Rank {rank:2d} | AP@50={trial.value:.4f} | {trial.params} → {fname}")

    # Fixed-name best config for easy --config reference in train.py
    best = top_k[0]
    best_config = dict(base_config)
    best_config.update(best.params)
    # Rebuild augmentation list honouring the boolean toggle flags, then drop the flags
    best_config["augmentations"] = _filter_augmentations(base_config, best.params)
    for flag in _bool_flags:
        best_config.pop(flag, None)
    best_config.pop("_optuna_trial_number", None)
    best_config.pop("_optuna_ap50", None)
    best_path = os.path.join(output_dir, "best_hparams.yaml")
    with open(best_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
    print(f"\nBest config (AP@50={best.value:.4f}) → {best_path}")


def main():
    args = parse_args()

    # Stagger workers to avoid simultaneous DB creation
    if args.worker_jitter > 0:
        time.sleep(args.worker_jitter)

    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    # Default storage: SQLite on project NFS root
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
    # MedianPruner: don't prune until 5 trials are complete and epoch >= 10
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

    # Summary
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\n=== Worker done | Completed: {len(completed)} | Pruned: {len(pruned)} ===")
    if study.best_trial:
        print(f"Best AP@50 across all workers: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

    save_configs(study, base_config, args.output_config_dir, args.save_top_k)


if __name__ == "__main__":
    main()
