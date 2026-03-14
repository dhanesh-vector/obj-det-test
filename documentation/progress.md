# Project Progress

## Accomplishments So Far

### 1. Project Initialization
- Initialized a Python 3.12 project using `uv`.
- Set up a virtual environment (`.venv`).
- Installed core dependencies: `torch`, `torchvision`, `numpy`, and `tqdm`.

### 2. Directory Structure
Created the following standard directories to organize the project:
- `model/`: For model definitions and architecture components.
- `data/`: For datasets and dataloaders.
- `utils/`: For utility scripts and helper functions.
- `results/`: For saving metrics, outputs, and generated files.
- `train/`: For training scripts and configurations.
- `inference/`: For inference and evaluation scripts.

### 3. Model Implementation
Integrated the **PP-YOLOE** object detection architecture using the source code from [Gaurav14cs17/YOLOE](https://github.com/Gaurav14cs17/YOLOE).

**Key Components Implemented in `model/`:**
- **Model Architecture:** Added the entire YOLOE architecture components:
  - Backbone: `CSPResNet`
  - Neck: `CustomCSPPAN`
  - Head: `PPYOLOEHead`
- **Assigners:** Added required target assignment modules (e.g., `ATSSAssigner`) in the `model/assigners/` subdirectory.
- **Loss Functions:** Extracted and implemented the precise `YOLOELoss` logic into a new `model/loss.py` file. This handles:
  - Classification loss (BCE)
  - Bounding box regression loss (Smooth L1 / IoU-based)
  - DFL (Distribution Focal Loss)
- **Training Wrapper:** Updated the `YOLOEWithLoss` class inside `model/yoloe.py` to seamlessly compute losses using the integrated `YOLOELoss` implementation out-of-the-box.

### 4. Advanced Loss Implementation 
Implemented an advanced  Loss variant in `model/pu_loss.py` to handle scenarios with extreme missing labels (e.g., only 10% labeled data).
- **Soft Sampling (Gradient Re-weighting):** Down-weights the penalty for "background" anchors that have a high objectness score, reducing the suppression of valid but unlabelled objects (controlled by parameter `gamma`).
- **Focal IoU (FIoU) Weighting:** Up-weights the classification loss for labeled objects when their localization IoU is high, promoting tighter bounding boxes (controlled by parameter `beta`).
- **Integration:** Updated `YOLOEWithLoss` in `model/yoloe.py` to optionally use the new `YOLOEPUFocalLoss` by passing `use_pu_loss=True`.

### 5. Training Pipeline & Evaluation Improvements
- **Training Script:** Implemented `train.py` with support for Baseline and PU-Loss modes.
- **Evaluation:** Added COCO mAP evaluation using `torchmetrics.detection.MeanAveragePrecision`. Suppressed the `warn_on_many_detections` warning in `utils/metrics.py` as 100 detections are sufficient for ultrasound.
- **Learning Rate Scheduler:** Added a linear warmup (3 epochs) followed by a Cosine Annealing decay scheduler to stabilize early training gradients and prevent overfitting.
- **Data Augmentation:** Integrated `albumentations` for image augmentations (Horizontal/Vertical Flip, ColorJitter) inside `data/dataset.py` with proper bounding box transformations.

### 6. Codebase Refactoring & Loss Fixes
- **Loss Refactoring:** Extracted target assignment logic (`_assign_targets`) to a reusable method in the base `YOLOELoss` class to prevent duplication in `YOLOEPUFocalLoss`.
- **Gradient Stability:** Clamped `combined_weight` in `pu_loss.py` to `max=50.0` to prevent extreme classification gradients when `num_pos` is very small relative to `num_anchors`.
- **Distribution Focal Loss (DFL):** Discovered that the regression head outputs a continuous distribution over 17 discrete bins (`reg_max=16`). Implemented and integrated the missing `DistributionFocalLoss` computation in both `loss.py` and `pu_loss.py` to ensure sharp and unimodal spatial distributions for accurate bounding box localization.

### 7. Results Analysis & Inference
- **Analysis:** Compared baseline vs PU-loss metrics. Found that while PU-loss achieved lower validation loss, it suffered from significantly worse mAP scores indicating an issue with actual object detection performance.
- **Visualization:** Created `inference/visualize.py` using `matplotlib` to render a 3x2 grid of randomly selected validation images comparing Ground Truth (green), Baseline predictions (red), and PU-loss predictions (yellow).

### 8. Training Stability Fixes & Augmentation Improvements

Analysis of training run `yoloe_train_650` revealed two critical bugs and several configuration weaknesses causing unstable mAP and diverging val loss.

**Bug Fixes (`train/train.py`):**
- **Validation BatchNorm corruption (critical):** The `validate()` function was toggling between `model.train()` and `model.eval()` on every batch inside `torch.no_grad()`. `no_grad` does not prevent BatchNorm from updating its running statistics in train mode, so each val batch was corrupting BN running mean/variance — the primary cause of the exploding, unstable val loss. Fixed by separating validate into two distinct passes: a train-mode loss pass, then an eval-mode prediction pass.
- **Missing gradient clipping:** No `clip_grad_norm_` was applied, allowing occasional gradient explosions (visible as the train loss spike at epoch 23). Added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)`.

**Early Stopping (`train/train.py`, `train/config/default.yaml`):**
- Added configurable early stopping based on `mAP@50` with `patience` and `min_delta` parameters.
- Config defaults: `early_stopping_patience: 10`, `early_stopping_min_delta: 0.001`. Set patience to `0` to disable.

**Augmentation Improvements (`train/config/default.yaml`):**
- Replaced generic `ColorJitter` (which included `saturation`/`hue`, meaningless for ultrasound) with ultrasound-appropriate intensity augmentations:
  - `RandomBrightnessContrast` — simulates gain/TGC variation across machines
  - `GaussianBlur` — simulates depth-dependent focus variation
  - `GaussNoise` — simulates electronic/quantisation noise
  - `CLAHE` — simulates contrast normalisation preprocessing differences
  - `RandomGamma` — simulates probe sensitivity differences
- All new augmentations are intensity-only; bounding boxes are unaffected. Spatial augmentations (flips) are already handled correctly via albumentations `BboxParams`.

### 9. Training Metric & Early Stopping Improvements (Runs 652–653)

**Early stopping criterion fixed (`train/train.py`):**
- Early stopping was previously keyed on `val_loss`, which diverged from epoch 2 onward even as AP@50 continued improving (peaked at 0.507 @ epoch 8 in run 652). This caused premature stopping at epoch 12.
- Switched early stopping to track **AP@50** directly. Counter resets only when AP improves by more than `early_stopping_min_delta` (default `0.001`). Checkpoint saving and early stopping counter are now unified in a single block.
- Removed `best_val_loss` from `training_info` (no longer the stopping signal).

**Precision & Recall tracking (`utils/metrics.py`):**
- `MeanAveragePrecision` (torchmetrics) returns `mar_100` (COCO-style max recall) but no per-threshold precision/recall.
- Added `_precision_recall_at_threshold()` to `Evaluator`: computes TP/FP/FN via greedy IoU matching at `conf=0.5`, `IoU=0.5` across the full validation set. Results logged per epoch and saved to JSON.
- This revealed that precision was **~1–3%** throughout run 653 (Prec=0.014 at best epoch) despite AP@50=0.43 — meaning ~68 false positives per true positive.

**Config changes (`train/config/default.yaml`):**
- `learning_rate`: `0.001` → `0.0005` (reduce peak LR to slow convergence and reduce val loss divergence)
- `weight_decay`: `0.0005` → `0.001` (stronger regularization)
- `early_stopping_min_delta`: `0.05` → `0.001` (AP units; previous value was tuned for val_loss scale)
- Added `ShiftScaleRotate` augmentation (`shift=0.05, scale=0.1, rotate=10°, p=0.4`) for geometric variation

### 10. Loss Normalization Fix — Root Cause of Low Precision (Run 654)

**Root cause identified in `model/loss.py`:**

The classification loss used `.mean()` for both positive and negative anchor sets:
```python
pos_loss = bce(pos_anchors).mean() * pos_weight   # ~13 anchors × 50
neg_loss = bce(neg_anchors).mean()                 # ~8000 anchors, averaged down
```
This created a ~30,000× per-anchor gradient imbalance: positive anchors received massive gradient to predict 1, while each background anchor received near-zero gradient to predict 0. The model learned to fire everywhere (high recall, ~1% precision).

**Fix (`model/loss.py`, `model/pu_loss.py`):**
- Replaced `.mean() * pos_weight` pattern with `.sum() / num_pos` for both positive and negative terms.
- This ensures the total gradient from all background anchors is commensurate with the total gradient from positive anchors — the standard normalization used in YOLO-family losses.
- `pu_loss.py` had the same bug and was fixed identically; the no-GT image case normalizes by `num_anchors` (equivalent to mean) since there are no positives to reference.

**Impact (run 654, epochs 2–7):**
- Precision jumped from ~0.014 (run 653) to **0.40–0.52** within the first few epochs
- AP@50 reached 0.404 by epoch 3 (previously took until epoch 10)
- Val loss converged more stably (remained in 28–34 range through epoch 5 vs immediate divergence before)

### 11. Label Smoothing (Run 655)

**Added `label_smooth` hyperparameter** (`model/loss.py`, `model/pu_loss.py`, `model/yoloe.py`, `train/train.py`, `train/config/default.yaml`):
- Positive classification targets changed from hard `1.0` to `1.0 - label_smooth` (default `0.1` → target `0.9`).
- Propagated through the full stack: `YOLOELoss._assign_targets` → `YOLOEWithLoss.__init__` → `train.py --label_smooth` arg → `default.yaml`.
- Rationale: prevents the model from being trained to maximum confidence on positives, improving calibration and generalization. Standard practice in modern YOLO variants.
- Dropout was considered but ruled out: this is an all-Conv+BN architecture where BatchNorm and Dropout interact destructively (BN normalizes the distribution that Dropout corrupts). Weight decay + augmentation + label smoothing are the correct regularization levers for this architecture.

### 12. Scan-Aware Slice Sampling (`data/sampler.py`)

**Problem:** Ultrasound volumes are acquired at very small inter-slice spacing, so adjacent slices are near-duplicates. With a standard shuffled DataLoader the model receives many redundant gradient updates from almost-identical images within the same epoch, increasing effective overfitting risk and slowing convergence on genuinely diverse samples.

The train/val split is already done at the volume level (no leakage), but redundancy *within* the training set is still high.

**Solution — `ScanAwareSampler`** (`data/sampler.py`):
- Parses filenames using the study convention `{ScanID}-{SliceNumber}.ext` (e.g. `CEM004A3-02275.png`) to group all slices by their scan ID (patient + sample + scan index).
- At the start of each epoch, divides each scan's sorted slice list into non-overlapping windows of `stride` consecutive frames and picks one slice at random from each window.
- Globally shuffles the resulting subset and yields those dataset indices — drop-in replacement for `shuffle=True`.
- Calls `set_epoch(epoch)` each epoch so the random offset varies, ensuring all slices are seen across epochs rather than always the same 1/stride fraction.

**Integration:**
- `train.py`: imports `ScanAwareSampler`; when `--slice_stride > 1` passes `sampler=` to `DataLoader` (disabling `shuffle`); calls `sampler.set_epoch(epoch)` at the top of each epoch loop.
- `train/config/default.yaml`: `slice_stride: 5` (retains ~20% of slices per epoch).
- `slice_stride: 1` disables the sampler entirely (falls back to standard shuffle).

**Stride chosen from dataset analysis:**
- 2306 total training slices across 13 scans forming **50 consecutive-slice runs**
- Median run length: **30 slices**, mean: 46, max: 225 (CEM009A1)
- 98%+ of adjacent included-slice pairs differ by exactly 1 frame — near-duplicates throughout

| stride | slices/epoch | % of total | slices per median-30 run |
|--------|-------------|------------|--------------------------|
| 5      | ~461        | 20%        | ~6                       |
| **10** | **~230**    | **10%**    | **~3**                   |
| 15     | ~154        | 7%         | ~2                       |

`slice_stride: 10` is the default — keeps ~3 representative samples per median run, reducing redundancy ~10× per epoch while maintaining full coverage across all 100 epochs.

### Next Steps
- Evaluate run 655 results (label smoothing + loss fix combined): target AP@50 > 0.50 with Prec > 0.40.
- If val loss continues diverging after epoch ~10, consider increasing `weight_decay` further or reducing `epochs` + relying on best-AP checkpoint.
- Pseudo-labeling on unlabeled data is a viable next step if AP plateaus — run best checkpoint on unlabeled images, keep high-confidence detections as new training labels, retrain.
