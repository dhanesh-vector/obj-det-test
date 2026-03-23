"""
Visualize all v5 augmentations with bbox overlay.

Covers every augmentation in the v5 search space:
  - HorizontalFlip / VerticalFlip       (always-on base)
  - CLAHE                               (tunable; shown at extreme clip_limit)
  - ShiftScaleRotate                    (tunable; shown at extremes)
  - RandomBrightnessContrast            (tunable; shown at both poles)
  - MedianBlur                          (tunable; shown at blur_limit=7)
  - SpeckleNoise / MultiplicativeNoise  (tunable; light / medium / heavy)
  - TissueAwareCopyPaste                (custom; p forced to 1.0)
  - Cross-scan Mosaic                   (shown as a separate row — needs 4 images)

Layout
------
  Rows 0..N_BASE-1 : N_BASE source images × (1 original + 8 single-aug) columns
  Row  N_BASE      : mosaic examples — 3 independently built mosaics

Usage
-----
    python test/test_aug_contrast.py [--n_images 3] [--data_dir PATH]
"""

import sys
import os
import argparse
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import albumentations as A
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import UltrasoundDataset, collate_fn
from data.augmentations import TissueAwareCopyPaste

DATA_DIR = '/projects/tenomix/ml-share/training/07/data'

# ─── Augmentation definitions (all forced p=1.0 for deterministic display) ───

SINGLE_AUG_VARIANTS = [
    # (column title, albumentations pipeline or None for custom)
    ("HFlip\n(always)", A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )),
    ("VFlip\n(always)", A.Compose(
        [A.VerticalFlip(p=1.0)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )),
    ("CLAHE\nclip=2.0 (training)", A.Compose(
        [A.CLAHE(clip_limit=(2.0, 2.0), tile_grid_size=(8, 8), p=1.0)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )),
    ("ShiftScaleRotate\nextreme", A.Compose(
        [A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.4, rotate_limit=45, p=1.0)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )),
    ("BC dark\n(−0.4/−0.4)", A.Compose(
        [A.RandomBrightnessContrast(
            brightness_limit=(-0.4, -0.4), contrast_limit=(-0.4, -0.4), p=1.0)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )),
    ("BC bright\n(+0.4/+0.4)", A.Compose(
        [A.RandomBrightnessContrast(
            brightness_limit=(0.4, 0.4), contrast_limit=(0.4, 0.4), p=1.0)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )),
    ("MedianBlur\nlimit=7", A.Compose(
        [A.MedianBlur(blur_limit=(7, 7), p=1.0)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )),
    ("Speckle\nheavy [0.80–1.20]", A.Compose(
        [A.MultiplicativeNoise(multiplier=(0.80, 1.20), p=1.0)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )),
    ("CopyPaste\n(p=1.0)", None),   # handled separately — custom augmentation
]

N_AUG_COLS = 1 + len(SINGLE_AUG_VARIANTS)   # original + variants


# ─── Helpers ─────────────────────────────────────────────────────────────────

def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Float [0,1] CHW tensor → uint8 HWC numpy."""
    img = tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img


def apply_albu(pipeline, img_np: np.ndarray, boxes: np.ndarray) -> tuple:
    """Run an albumentations Compose pipeline. Returns (img_np, boxes_np)."""
    result = pipeline(image=img_np, bboxes=boxes.tolist(), labels=[0] * len(boxes))
    out_img = result['image']
    out_boxes = np.array(result['bboxes'], dtype=np.float32).reshape(-1, 4) \
        if result['bboxes'] else np.empty((0, 4), dtype=np.float32)
    return out_img, out_boxes


def draw_boxes(ax, img_np: np.ndarray, boxes: np.ndarray,
               color: str = 'red', lw: float = 1.5, title: str = ''):
    """Imshow image and draw bounding boxes."""
    display = img_np[:, :, 0] if img_np.shape[2] == 1 else img_np
    ax.imshow(display, cmap='gray', vmin=0, vmax=255, aspect='auto')
    for x1, y1, x2, y2 in boxes:
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=lw, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
    ax.set_title(title, fontsize=6.5, pad=2)
    ax.axis('off')


# ─── Copy-paste helper ────────────────────────────────────────────────────────

_copy_paste_aug = TissueAwareCopyPaste(
    p=1.0, n_candidates=50, std_tol=30.0, radius=30,
    min_box_area=256, max_pastes=2, tissue_threshold=10
)


def apply_copy_paste(img_np: np.ndarray, boxes: np.ndarray) -> tuple:
    labels = np.zeros(len(boxes), dtype=np.int64)
    out_img, out_boxes, _ = _copy_paste_aug(img_np.copy(), boxes.copy(), labels)
    return out_img, out_boxes


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n_images', type=int, default=3,
                        help='Number of source images for the single-aug rows (default: 3)')
    parser.add_argument('--data_dir', default=DATA_DIR)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    dataset = UltrasoundDataset(root_dir=args.data_dir, split='train', mosaic_prob=0.0)
    if len(dataset) == 0:
        print("Dataset is empty — check --data_dir.")
        return

    print(f"Dataset: {len(dataset)} training images.")

    # Pick source images at even spacing; prefer images that have at least one GT box
    indices_with_boxes = [
        i for i in range(len(dataset)) if len(dataset[i][1]['boxes']) > 0
    ]
    step = max(1, len(indices_with_boxes) // args.n_images)
    src_indices = indices_with_boxes[::step][:args.n_images]
    samples = [dataset[i] for i in src_indices]
    print(f"Source indices: {src_indices}")

    # ── Build figure ─────────────────────────────────────────────────────────
    n_single_rows = len(samples)
    n_mosaic_cols = 3
    n_total_rows = n_single_rows + 1      # +1 for mosaic row

    cell_w, cell_h = 2.2, 2.2
    fig_w = max(N_AUG_COLS * cell_w, 20)
    fig_h = n_total_rows * cell_h + 0.6   # extra for row labels

    fig = plt.figure(figsize=(fig_w, fig_h))

    # GridSpec: single-aug rows use N_AUG_COLS columns;
    # mosaic row uses n_mosaic_cols cells (left-aligned)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        n_total_rows, N_AUG_COLS,
        figure=fig, hspace=0.35, wspace=0.04,
        top=0.93, bottom=0.03, left=0.02, right=0.98,
    )

    # ── Single-augmentation rows ──────────────────────────────────────────────
    col_titles = ["Original"] + [title for title, _ in SINGLE_AUG_VARIANTS]

    for row, (img_tensor, target) in enumerate(samples):
        img_np   = tensor_to_uint8(img_tensor)
        boxes_np = target['boxes'].numpy()

        ax_col0 = None
        for col, (title, pipeline) in enumerate(
                [("Original", None)] + SINGLE_AUG_VARIANTS):

            ax = fig.add_subplot(gs[row, col])
            if col == 0:
                ax_col0 = ax

            if pipeline is None and title.startswith("CopyPaste"):
                aug_img, aug_boxes = apply_copy_paste(img_np.copy(), boxes_np.copy())
            elif pipeline is None:
                aug_img, aug_boxes = img_np.copy(), boxes_np.copy()
            else:
                aug_img, aug_boxes = apply_albu(pipeline, img_np.copy(), boxes_np.copy())

            box_color = 'lime' if len(aug_boxes) > len(boxes_np) else 'red'
            draw_boxes(ax, aug_img, aug_boxes, color=box_color,
                       title=col_titles[col] if row == 0 else '')

        # Row label — reuse the already-drawn col-0 axes (avoids clobbering it)
        ax_col0.set_ylabel(
            f"Image {row + 1}", fontsize=7, rotation=0,
            labelpad=40, va='center'
        )

    # ── Mosaic row ────────────────────────────────────────────────────────────
    # Build mosaics using the dataset's built-in _load_mosaic, then display side-by-side
    mosaic_dataset = UltrasoundDataset(
        root_dir=args.data_dir, split='train', mosaic_prob=1.0
    )

    # Column header spanning first n_mosaic_cols cells
    ax_hdr = fig.add_subplot(gs[n_single_rows, 0])
    ax_hdr.set_title("Cross-scan Mosaic (3 examples)", fontsize=8,
                      loc='left', pad=3, color='steelblue', fontweight='bold')
    ax_hdr.axis('off')

    mosaic_indices = random.sample(range(len(mosaic_dataset)), k=min(n_mosaic_cols, len(mosaic_dataset)))
    for m_col, m_idx in enumerate(mosaic_indices):
        img_tensor, target = mosaic_dataset[m_idx]
        img_np   = tensor_to_uint8(img_tensor)
        boxes_np = target['boxes'].numpy()
        ax = fig.add_subplot(gs[n_single_rows, m_col])
        draw_boxes(ax, img_np, boxes_np, color='cyan', lw=1.2,
                   title=f"Mosaic {m_col + 1}")

    # Hide unused cells in mosaic row
    for m_col in range(n_mosaic_cols, N_AUG_COLS):
        fig.add_subplot(gs[n_single_rows, m_col]).axis('off')

    # ── Legend & suptitle ────────────────────────────────────────────────────
    fig.suptitle(
        "v5 augmentations at extremes  —  red=GT boxes, lime=copy-paste additions, cyan=mosaic GT",
        fontsize=9, y=0.975
    )

    os.makedirs('results', exist_ok=True)
    save_path = 'results/aug_v5_overview.png'
    fig.savefig(save_path, bbox_inches='tight', dpi=140)
    plt.close(fig)
    print(f"Saved → {save_path}")


if __name__ == '__main__':
    main()
