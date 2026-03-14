"""
Visualize the combined effect of contrast-altering augmentations at their extremes.

Focuses on the three augmentations that all affect perceived brightness/contrast:
  - CLAHE
  - RandomGamma
  - RandomBrightnessContrast

Shows each individually and in combination, forcing extreme parameter values so
any destructive interactions are visible. Saves a grid to results/.

Usage:
    python test/test_aug_contrast.py [--n_images 4]
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import albumentations as A
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import UltrasoundDataset, collate_fn

DATA_DIR = '/projects/tenomix/ml-share/training/07/data'

# ---------------------------------------------------------------------------
# Augmentation variants — always_apply so we don't rely on randomness
# ---------------------------------------------------------------------------
CLAHE     = A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)
GAMMA_LO  = A.RandomGamma(gamma_limit=(80, 80),   p=1.0)   # darkens midtones
GAMMA_HI  = A.RandomGamma(gamma_limit=(120, 120), p=1.0)   # brightens midtones
BC_LO     = A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.3), contrast_limit=(-0.3, -0.3), p=1.0)
BC_HI     = A.RandomBrightnessContrast(brightness_limit=( 0.3,  0.3), contrast_limit=( 0.3,  0.3), p=1.0)

def make_pipeline(*transforms):
    return A.Compose(list(transforms), bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

VARIANTS = [
    ("Original",           make_pipeline()),
    ("CLAHE",              make_pipeline(CLAHE)),
    ("Gamma↓ (80)",        make_pipeline(GAMMA_LO)),
    ("Gamma↑ (120)",       make_pipeline(GAMMA_HI)),
    ("BC↓ (-0.3)",         make_pipeline(BC_LO)),
    ("BC↑ (+0.3)",         make_pipeline(BC_HI)),
    ("CLAHE+Gamma↓",       make_pipeline(CLAHE, GAMMA_LO)),
    ("CLAHE+Gamma↑",       make_pipeline(CLAHE, GAMMA_HI)),
    ("CLAHE+BC↓",          make_pipeline(CLAHE, BC_LO)),
    ("CLAHE+BC↑",          make_pipeline(CLAHE, BC_HI)),
    ("Gamma↓+BC↓",         make_pipeline(GAMMA_LO, BC_LO)),
    ("Gamma↑+BC↑",         make_pipeline(GAMMA_HI, BC_HI)),
    ("All extremes↓",      make_pipeline(CLAHE, GAMMA_LO, BC_LO)),
    ("All extremes↑",      make_pipeline(CLAHE, GAMMA_HI, BC_HI)),
]

N_COLS = len(VARIANTS)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tensor_to_uint8_numpy(tensor):
    """Convert float [0,1] CHW tensor → uint8 HWC numpy array."""
    img = tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    # Ensure 3-channel for albumentations
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img

def apply_aug(pipeline, img_np, boxes_np, labels):
    """Apply an albumentations pipeline. Returns uint8 HWC numpy array."""
    result = pipeline(image=img_np, bboxes=boxes_np.tolist(), labels=labels)
    return result['image']

def draw_boxes(ax, img_np, boxes_np, color='red', lw=1.5):
    ax.imshow(img_np if img_np.shape[2] == 3 else img_np[:, :, 0], cmap='gray', vmin=0, vmax=255)
    h, w = img_np.shape[:2]
    for box in boxes_np:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              linewidth=lw, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    ax.axis('off')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_images', type=int, default=4, help='Number of images to visualize')
    args = parser.parse_args()

    dataset = UltrasoundDataset(root_dir=DATA_DIR, split='train')
    if len(dataset) == 0:
        print("Dataset is empty — check DATA_DIR.")
        return
    print(f"Dataset: {len(dataset)} training images. Sampling {args.n_images}.")

    # Grab a fixed sample (not random, so the grid is reproducible)
    indices = list(range(0, len(dataset), max(1, len(dataset) // args.n_images)))[:args.n_images]
    samples = [dataset[i] for i in indices]

    n_rows = len(samples)
    fig_w = max(N_COLS * 2.2, 20)
    fig_h = n_rows * 2.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(n_rows, N_COLS, figure=fig, hspace=0.05, wspace=0.03)

    for row, (img_tensor, target) in enumerate(samples):
        img_np   = tensor_to_uint8_numpy(img_tensor)
        boxes_np = target['boxes'].numpy()        # (N, 4) xyxy
        labels   = target['labels'].tolist()

        for col, (title, pipeline) in enumerate(VARIANTS):
            aug_img = apply_aug(pipeline, img_np.copy(), boxes_np, labels)
            ax = fig.add_subplot(gs[row, col])
            draw_boxes(ax, aug_img, boxes_np)
            if row == 0:
                ax.set_title(title, fontsize=7, pad=3)

    fig.suptitle(
        "Contrast augmentation interactions at extremes  (red boxes = GT, fixed per row)",
        fontsize=10, y=1.01
    )

    os.makedirs('results', exist_ok=True)
    save_path = 'results/aug_contrast_extremes.png'
    fig.savefig(save_path, bbox_inches='tight', dpi=130)
    plt.close(fig)
    print(f"Saved → {save_path}")

if __name__ == '__main__':
    main()
