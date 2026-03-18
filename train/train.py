import sys
import os
import json
import time
import random
from datetime import datetime
import configargparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.yoloe import build_yoloe, YOLOEWithLoss
from data.dataset import UltrasoundDataset, collate_fn
from data.sampler import ScanAwareSampler
from utils.metrics import decode_predictions, Evaluator
from utils.plots import plot_validation_summary

def parse_args():
    parser = configargparse.ArgParser(default_config_files=['train/config/default.yaml'], ignore_unknown_config_file_keys=True)
    parser.add('-c', '--config', is_config_file=True, help='Path to config file')
    parser.add('--data_dir', type=str, default='/projects/tenomix/ml-share/training/07/data', help='Dataset root directory')
    parser.add('--model_size', type=str, default='s', choices=['s', 'm', 'l', 'x'], help='YOLOE model size')
    parser.add('--num_classes', type=int, default=1, help='Number of classes')
    parser.add('--batch_size', type=int, default=16, help='Training batch size')
    parser.add('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add('--num_workers', type=int, default=8, help='Number of DataLoader workers')
    parser.add('--use_pu_loss', action='store_true', help='Use PU Focal Loss')
    parser.add('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
    parser.add('--checkpoints_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add('--results_dir', type=str, default='results', help='Directory to save results')
    parser.add('--early_stopping_patience', type=int, default=20, help='Epochs without EMA SliceAP@50 improvement before stopping (0 to disable)')
    parser.add('--early_stopping_min_delta', type=float, default=0.005, help='Minimum EMA SliceAP@50 increase to count as progress')
    parser.add('--ema_alpha', type=float, default=0.3, help='EMA smoothing factor for SliceAP@50 used as early-stopping criterion (matches tune_hyperparams_v5.py)')
    parser.add('--label_smooth', type=float, default=0.0, help='Label smoothing for positive targets (e.g. 0.1 sets target to 0.9)')
    parser.add('--slice_stride', type=int, default=1, help='ScanAwareSampler window size: sample 1 slice per N consecutive slices per scan per epoch. 1 = all slices (standard shuffle). Recommended: 5-10 for dense ultrasound volumes.')
    parser.add('--mosaic_prob', type=float, default=0.0, help='Probability of replacing a training sample with a 2×2 cross-scan mosaic. 0.0 disables mosaic (default). Recommended: 0.5.')
    parser.add('--seed', type=int, default=42, help='Global random seed for reproducibility.')
    parser.add('--warmup_epochs', type=int, default=8, help='Number of linear warmup epochs before cosine decay.')
    parser.add('--use_median_blur', action='store_true', help='Add MedianBlur augmentation (speckle-aware, better than GaussianBlur for ultrasound).')
    parser.add('--median_blur_limit', type=int, default=5, help='Max kernel size for MedianBlur (must be odd). Default: 5.')
    parser.add('--median_blur_p', type=float, default=0.3, help='Probability of applying MedianBlur. Default: 0.3.')
    parser.add('--use_multiplicative_noise', action='store_true', help='Add MultiplicativeNoise augmentation (simulates ultrasound speckle noise).')
    parser.add('--multiplicative_noise_multiplier_min', type=float, default=0.9, help='Lower bound of multiplicative noise multiplier range. Default: 0.9.')
    parser.add('--multiplicative_noise_multiplier_max', type=float, default=1.1, help='Upper bound of multiplicative noise multiplier range. Default: 1.1.')
    parser.add('--multiplicative_noise_p', type=float, default=0.4, help='Probability of applying MultiplicativeNoise. Default: 0.4.')

    return parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Enforce determinism for all CUDA ops (scatter, index_add, upsample, etc.)
    # Set warn_only=True to surface any non-deterministic ops without crashing.
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # Seed albumentations 2.x internal RNGs (uses its own numpy Generator + random.Random,
    # not the global random state seeded above).
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if hasattr(dataset, 'transform') and dataset.transform is not None:
            dataset.transform.set_random_seed(worker_seed)


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        
        # Move targets to device
        device_targets = []
        for target in targets:
            device_target = {k: v.to(device) for k, v in target.items()}
            device_targets.append(device_target)
            
        optimizer.zero_grad()
        loss_dict = model(images, device_targets)
        
        loss = loss_dict['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    total_loss = 0.0
    evaluator = Evaluator()

    # Loss pass: train mode so BN uses batch stats (consistent with training),
    # but we must NOT intermix eval mode here to avoid corrupting running stats.
    model.train()
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            device_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, device_targets)
            total_loss += loss_dict['loss'].item()

    # Prediction pass: eval mode so BN uses stable running stats.
    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            device_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            cls_scores, reg_dists = model(images)

            h, w = images.shape[2:]
            feat_sizes = [(h // s, w // s) for s in model.model.strides]
            anchor_points, stride_tensor = model.model.get_anchor_points(feat_sizes, device, images.dtype)
            batch_preds = decode_predictions(cls_scores, reg_dists, anchor_points, stride_tensor)
            evaluator.update(batch_preds, device_targets)

    parsed_results = evaluator.compute()
    return total_loss / len(dataloader), parsed_results, evaluator

def main():
    args = parse_args()
    set_seed(args.seed)

    # Load raw config to get augmentations
    import yaml
    import albumentations as A
    with open(args.config if args.config else 'train/config/default.yaml', 'r') as f:
        raw_config = yaml.safe_load(f)

    aug_list = []
    if 'augmentations' in raw_config:
        for aug in raw_config['augmentations']:
            for name, params in aug.items():
                aug_class = getattr(A, name)
                aug_list.append(aug_class(**params))

    if args.use_median_blur:
        aug_list.append(A.MedianBlur(blur_limit=args.median_blur_limit, p=args.median_blur_p))
    if args.use_multiplicative_noise:
        aug_list.append(A.MultiplicativeNoise(
            multiplier=(args.multiplicative_noise_multiplier_min, args.multiplicative_noise_multiplier_max),
            p=args.multiplicative_noise_p
        ))

    train_transform = None
    if aug_list:
        train_transform = A.Compose(
            aug_list,
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
        )

    copy_paste = None
    copy_paste_cfg = raw_config.get('copy_paste', {})
    if copy_paste_cfg.get('enabled', False):
        from data.augmentations import TissueAwareCopyPaste
        copy_paste_params = {k: v for k, v in copy_paste_cfg.items() if k != 'enabled'}
        copy_paste = TissueAwareCopyPaste(**copy_paste_params)
        print(f"TissueAwareCopyPaste enabled: {copy_paste_params}")

    # Setup directories
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Initialize Dataset and DataLoader
    print("Loading datasets...")
    train_dataset = UltrasoundDataset(root_dir=args.data_dir, split='train', transform=train_transform,
                                       mosaic_prob=args.mosaic_prob, copy_paste=copy_paste)
    val_dataset = UltrasoundDataset(root_dir=args.data_dir, split='val')    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    if args.slice_stride > 1:
        train_sampler = ScanAwareSampler(train_dataset, stride=args.slice_stride, seed=args.seed)
    else:
        train_sampler = None

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),  # mutually exclusive with sampler
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if args.device == 'cuda' else False,
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # Initialize Model
    print(f"Building YOLOE model (size: {args.model_size})...")
    base_model = build_yoloe(model_size=args.model_size, num_classes=args.num_classes)
    model = YOLOEWithLoss(
        model=base_model,
        num_classes=args.num_classes,
        use_pu_loss=args.use_pu_loss,
        label_smooth=args.label_smooth
    ).to(device)
    
    # Setup Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # Setup Learning Rate Scheduler (warmup_epochs linear warmup + Cosine Decay)
    warmup_epochs = args.warmup_epochs
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(args.epochs - warmup_epochs), eta_min=1e-5
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    # Track metrics for JSON report
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_file = os.path.join(args.results_dir, f"training_run_{run_timestamp}.json")
    best_checkpoint_path = os.path.join(args.checkpoints_dir, f"best_model_{run_timestamp}.pth")
    
    training_info = {
        "timestamp_start": datetime.now().isoformat(),
        "args": vars(args),
        "seed": args.seed,
        "metrics": {
            "train_loss": [],
            "val_loss": [],
            "map_35": [],
            "map_50": [],
            "map_75": [],
            "map": [],
            "mar_100": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "slice_ap_50": [],
            "slice_ap_75": [],
            "ema_slice_ap_50": [],
            "precision_at_recall80": [],
            "recall_at_precision80": [],
            "epoch_times": []
        },
        "best_epoch": -1,
        "best_ap_50": -1.0,
        "best_slice_ap_ema": -1.0,
        "checkpoint_path": best_checkpoint_path
    }
    
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Early-stopping criterion: EMA SliceAP@50 (alpha={args.ema_alpha}, "
          f"patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta})")
    start_train_time = time.time()

    early_stopping_counter = 0
    ema_slice_ap = None   # initialised on first epoch (same as tune_hyperparams_v5.py)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train and Validate
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, map_metrics, last_evaluator = validate(model, val_loader, device)

        # EMA of SliceAP@50 — matches tune_hyperparams_v5.py exactly
        raw_slice_ap = map_metrics['slice_ap_50']
        if ema_slice_ap is None:
            ema_slice_ap = raw_slice_ap
        else:
            ema_slice_ap = args.ema_alpha * raw_slice_ap + (1.0 - args.ema_alpha) * ema_slice_ap

        # Step the learning rate scheduler
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        epoch_time = time.time() - epoch_start

        print(f"Epoch [{epoch}/{args.epochs}] Time: {epoch_time:.2f}s | LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  mAP@35: {map_metrics['map_35']:.4f} | mAP@50: {map_metrics['map_50']:.4f} | "
              f"mAP@75: {map_metrics['map_75']:.4f} | mAP@50:95: {map_metrics['map']:.4f}")
        print(f"  SliceAP@50: {raw_slice_ap:.4f} | EMA_SliceAP@50: {ema_slice_ap:.4f} | "
              f"SliceAP@75: {map_metrics['slice_ap_75']:.4f}")
        print(f"  Prec: {map_metrics['precision']:.4f} | Recall: {map_metrics['recall']:.4f} | "
              f"F1: {map_metrics['f1']:.4f} | Recall@100: {map_metrics['mar_100']:.4f}")
        print(f"  P@R=0.8: {map_metrics['precision_at_recall80']:.4f} | "
              f"R@P=0.8: {map_metrics['recall_at_precision80']:.4f}")

        # Record metrics
        training_info["metrics"]["train_loss"].append(train_loss)
        training_info["metrics"]["val_loss"].append(val_loss)
        training_info["metrics"]["map_35"].append(map_metrics['map_35'])
        training_info["metrics"]["map_50"].append(map_metrics['map_50'])
        training_info["metrics"]["map_75"].append(map_metrics['map_75'])
        training_info["metrics"]["map"].append(map_metrics['map'])
        training_info["metrics"]["mar_100"].append(map_metrics['mar_100'])
        training_info["metrics"]["precision"].append(map_metrics['precision'])
        training_info["metrics"]["recall"].append(map_metrics['recall'])
        training_info["metrics"]["f1"].append(map_metrics['f1'])
        training_info["metrics"]["slice_ap_50"].append(raw_slice_ap)
        training_info["metrics"]["ema_slice_ap_50"].append(ema_slice_ap)
        training_info["metrics"]["slice_ap_75"].append(map_metrics['slice_ap_75'])
        training_info["metrics"]["precision_at_recall80"].append(map_metrics['precision_at_recall80'])
        training_info["metrics"]["recall_at_precision80"].append(map_metrics['recall_at_precision80'])
        training_info["metrics"]["epoch_times"].append(epoch_time)

        # Save best model and drive early stopping — both keyed on EMA SliceAP@50
        # (consistent with tune_hyperparams_v5.py)
        if ema_slice_ap > training_info["best_slice_ap_ema"] + args.early_stopping_min_delta:
            training_info["best_slice_ap_ema"] = ema_slice_ap
            training_info["best_ap_50"] = map_metrics['map_50']
            training_info["best_epoch"] = epoch
            early_stopping_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'map_35': map_metrics['map_35'],
                'ap_50': map_metrics['map_50'],
                'ap_75': map_metrics['map_75'],
                'map': map_metrics['map'],
                'slice_ap_50': raw_slice_ap,
                'ema_slice_ap_50': ema_slice_ap,
                'slice_ap_75': map_metrics['slice_ap_75'],
                'precision': map_metrics['precision'],
                'recall': map_metrics['recall'],
                'f1': map_metrics['f1'],
                'recall_100': map_metrics['mar_100'],
                'precision_at_recall80': map_metrics['precision_at_recall80'],
                'recall_at_precision80': map_metrics['recall_at_precision80'],
                'config': vars(args)
            }
            torch.save(checkpoint, best_checkpoint_path)
            print(f"--> Saved new best model  EMA SliceAP@50: {ema_slice_ap:.4f} "
                  f"| AP@50: {map_metrics['map_50']:.4f} (Val Loss: {val_loss:.4f})")
        else:
            early_stopping_counter += 1
            if args.early_stopping_patience > 0 and early_stopping_counter >= args.early_stopping_patience:
                print(f"Early stopping triggered: EMA SliceAP@50 did not improve by "
                      f"{args.early_stopping_min_delta} for {args.early_stopping_patience} epochs.")
                with open(progress_file, 'w') as f:
                    json.dump(training_info, f, indent=4)
                break

        # Periodically save the progress file
        with open(progress_file, 'w') as f:
            json.dump(training_info, f, indent=4)

    total_time = time.time() - start_train_time
    training_info["timestamp_end"] = datetime.now().isoformat()
    training_info["total_training_time_seconds"] = total_time
    
    # Final save of the progress file
    with open(progress_file, 'w') as f:
        json.dump(training_info, f, indent=4)

    # Save validation diagnostic plots using the last epoch's evaluator
    plots_path = progress_file.replace('.json', '_plots.png')
    plot_validation_summary(last_evaluator, training_info, plots_path)

    print("Training complete!")
    print(f"Best EMA SliceAP@50: {training_info['best_slice_ap_ema']:.4f} at Epoch {training_info['best_epoch']}")
    print(f"Corresponding AP@50: {training_info['best_ap_50']:.4f}")
    print(f"Run summary saved to {progress_file}")
    
if __name__ == "__main__":
    main()
