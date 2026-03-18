import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model.yoloe import build_yoloe, YOLOEWithLoss
from data.dataset import UltrasoundDataset
from utils.metrics import decode_predictions


def load_model(weights_path, use_pu_loss, device):
    base_model = build_yoloe(model_size='s', num_classes=1)
    model = YOLOEWithLoss(model=base_model, num_classes=1, use_pu_loss=use_pu_loss)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state = checkpoint['model_state_dict']
    # Checkpoints may store base model weights or full wrapper weights
    try:
        model.model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def get_predictions(model, img_tensor, device):
    with torch.no_grad():
        cls_scores, reg_dists = model(img_tensor)
        feat_sizes = [(img_tensor.shape[2] // s, img_tensor.shape[3] // s)
                      for s in model.model.strides]
        anchor_points, stride_tensor = model.model.get_anchor_points(
            feat_sizes, device, img_tensor.dtype
        )
        return decode_predictions(cls_scores, reg_dists, anchor_points, stride_tensor)[0]


def draw_boxes(ax, boxes, color, linestyle='-', label=None):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, linestyle=linestyle,
            facecolor='none', label=label if i == 0 else None,
        )
        ax.add_patch(rect)


def visualize_merged(model, dataset, device, out_path, n_images=6):
    """Show GT vs merged-model predictions for n random val images."""
    indices = random.sample(range(len(dataset)), min(n_images, len(dataset)))

    cols = 2
    rows = (len(indices) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        image, target = dataset[idx]
        img_tensor = image.unsqueeze(0).to(device)
        preds = get_predictions(model, img_tensor, device)

        ax = axes[i]
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())

        draw_boxes(ax, target['boxes'], color='green', label='GT')
        if len(preds['boxes']) > 0:
            draw_boxes(ax, preds['boxes'][:1], color='royalblue', label='SWA Merged')

        ax.axis('off')

    # Hide any unused subplots
    for j in range(len(indices), len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=3, fontsize=12)
    fig.suptitle('SWA Merged Model — Validation Predictions', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization → {out_path}")


def visualize_comparison(model_baseline, model_pu, dataset, device, out_path, n_images=6):
    """Original mode: baseline (red) vs PU loss (yellow) vs GT (green)."""
    indices = random.sample(range(len(dataset)), min(n_images, len(dataset)))

    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        image, target = dataset[idx]
        img_tensor = image.unsqueeze(0).to(device)

        preds_b = get_predictions(model_baseline, img_tensor, device)
        preds_p = get_predictions(model_pu, img_tensor, device)

        ax = axes[i]
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())

        draw_boxes(ax, target['boxes'], color='green', label='GT')
        if len(preds_b['boxes']) > 0:
            draw_boxes(ax, preds_b['boxes'][:1], color='red', label='Baseline')
        if len(preds_p['boxes']) > 0:
            draw_boxes(ax, preds_p['boxes'][:1], color='yellow', linestyle='--', label='PU Loss')

        ax.axis('off')

    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(out_path)
    print(f"Saved visualization → {out_path}")


def visualize_best_vs_merged(model_best, model_merged, dataset, device, out_path, n_images=6):
    """
    3-column grid per image: GT boxes | best single checkpoint | SWA merged.
    Each row is one random val image.
    """
    indices = random.sample(range(len(dataset)), min(n_images, len(dataset)))

    fig, axes = plt.subplots(len(indices), 3, figsize=(18, 5 * len(indices)))
    if len(indices) == 1:
        axes = axes[None, :]  # ensure 2-D indexing

    col_titles = ['Ground Truth', 'Best Single Checkpoint', 'SWA Merged']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight='bold', pad=8)

    for row, idx in enumerate(indices):
        image, target = dataset[idx]
        img_tensor = image.unsqueeze(0).to(device)
        img_np = image.permute(1, 2, 0).cpu().numpy()

        preds_best   = get_predictions(model_best,   img_tensor, device)
        preds_merged = get_predictions(model_merged, img_tensor, device)

        for col in range(3):
            ax = axes[row, col]
            ax.imshow(img_np)
            ax.axis('off')

            # GT always shown in green on all columns
            draw_boxes(ax, target['boxes'], color='limegreen', label='GT')

            if col == 1 and len(preds_best['boxes']) > 0:
                draw_boxes(ax, preds_best['boxes'][:1], color='tomato', label='Best single')
            if col == 2 and len(preds_merged['boxes']) > 0:
                draw_boxes(ax, preds_merged['boxes'][:1], color='royalblue', label='SWA merged')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle('Best Single Checkpoint vs SWA Merged — Validation Set', fontsize=14)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization → {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions on val set')
    parser.add_argument('--merged-model', default=None,
                        help='Path to a merged/SWA .pt checkpoint')
    parser.add_argument('--best-model', default=None,
                        help='Path to the best single .pt checkpoint — '
                             'when combined with --merged-model shows a comparison grid')
    parser.add_argument('--data-dir', default='/projects/tenomix/ml-share/training/07/data',
                        help='Dataset root directory')
    parser.add_argument('--n-images', type=int, default=6,
                        help='Number of random val images to visualize')
    parser.add_argument('--out', default=None,
                        help='Output image path (defaults to inference/results_*.png)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset = UltrasoundDataset(root_dir=args.data_dir, split='val')

    if args.best_model and args.merged_model:
        # ── Comparison mode: best single vs SWA merged ─────────────────────
        print(f"Loading best single checkpoint: {args.best_model}")
        model_best = load_model(args.best_model, use_pu_loss=False, device=device)
        print(f"Loading merged model: {args.merged_model}")
        model_merged = load_model(args.merged_model, use_pu_loss=False, device=device)

        out_path = args.out or os.path.join(base_dir, 'inference', 'results_best_vs_merged.png')
        visualize_best_vs_merged(model_best, model_merged, dataset, device,
                                 out_path, n_images=args.n_images)

    elif args.merged_model:
        # ── Single merged-model mode ───────────────────────────────────────
        print(f"Loading merged model: {args.merged_model}")
        model = load_model(args.merged_model, use_pu_loss=False, device=device)

        out_path = args.out or os.path.join(
            base_dir, 'inference',
            f"results_merged_{os.path.splitext(os.path.basename(args.merged_model))[0]}.png"
        )
        visualize_merged(model, dataset, device, out_path, n_images=args.n_images)

    else:
        # ── Original comparison mode (baseline vs PU loss) ─────────────────
        baseline_weights = os.path.join(base_dir, 'checkpoints', 'best_model_20260315_101545.pth')
        pu_weights = os.path.join(base_dir, 'checkpoints', 'best_model_20260314_170531.pth')

        print("Loading models...")
        model_baseline = load_model(baseline_weights, use_pu_loss=False, device=device)
        model_pu = load_model(pu_weights, use_pu_loss=True, device=device)

        out_path = args.out or os.path.join(base_dir, 'inference', 'results_grid.png')
        visualize_comparison(model_baseline, model_pu, dataset, device, out_path,
                             n_images=args.n_images)


if __name__ == '__main__':
    main()
