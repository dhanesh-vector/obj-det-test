#!/usr/bin/env python3
"""
Create a 4-column × 3-row image grid.
Each row = one patient sample; each column = a consecutive frame from that sample.
Images are resized to half resolution before compositing.
"""
import os
import re
from pathlib import Path
from PIL import Image

IMAGE_DIR = Path("/projects/tenomix/ml-share/training/07/data/images/train")
OUTPUT_PATH = Path("/fs01/home/dhaneshr/code/obj-det-test/inference/sequence_grid.png")

N_COLS = 4   # consecutive frames per patient
N_ROWS = 3   # number of patients

# Pick 3 patients and a run of 4 consecutive frames each.
# Adjust these to taste.
SAMPLES = [
    "CEM012A1",
    "CEM007A1",
    "CEM005A1",
]


def get_sorted_frames(patient: str) -> list[Path]:
    """Return all frames for a patient, sorted by frame number."""
    files = sorted(
        IMAGE_DIR.glob(f"{patient}-*.png"),
        key=lambda p: int(re.search(r"-(\d+)\.png$", p.name).group(1)),
    )
    return files


def pick_consecutive(frames: list[Path], n: int, start_idx: int = 0) -> list[Path]:
    """Return n consecutive frames starting at start_idx."""
    if start_idx + n > len(frames):
        raise ValueError(
            f"Not enough frames: need {n} from index {start_idx}, "
            f"only {len(frames)} available."
        )
    return frames[start_idx : start_idx + n]


def main():
    rows: list[list[Path]] = []
    for patient in SAMPLES:
        frames = get_sorted_frames(patient)
        if len(frames) < N_COLS:
            raise RuntimeError(f"{patient}: only {len(frames)} frames, need {N_COLS}")
        # Pick from the middle of the sequence for visual interest
        mid = max(0, len(frames) // 2 - N_COLS // 2)
        chosen = pick_consecutive(frames, N_COLS, start_idx=mid)
        print(f"{patient}: using frames {[f.name for f in chosen]}")
        rows.append(chosen)

    # Load all images and resize to half resolution
    images: list[list[Image.Image]] = []
    for row_paths in rows:
        row_imgs = []
        for p in row_paths:
            img = Image.open(p).convert("RGB")
            half_w = img.width // 2
            half_h = img.height // 2
            row_imgs.append(img.resize((half_w, half_h), Image.LANCZOS))
        images.append(row_imgs)

    # All cells share the same size (use the first image as reference)
    cell_w, cell_h = images[0][0].size
    pad = 4  # pixels between cells

    grid_w = N_COLS * cell_w + (N_COLS - 1) * pad
    grid_h = N_ROWS * cell_h + (N_ROWS - 1) * pad

    grid = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))

    for r, row_imgs in enumerate(images):
        for c, img in enumerate(row_imgs):
            # Resize to cell size in case images differ across patients
            img = img.resize((cell_w, cell_h), Image.LANCZOS)
            x = c * (cell_w + pad)
            y = r * (cell_h + pad)
            grid.paste(img, (x, y))

    grid.save(OUTPUT_PATH)
    print(f"\nSaved grid ({grid_w}×{grid_h} px) → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
