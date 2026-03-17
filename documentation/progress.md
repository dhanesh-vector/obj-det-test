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

### 13. Cross-Scan Mosaic Augmentation (`data/dataset.py`)

**Motivation:** With only ~230 slices/epoch (after scan-aware subsampling), the model sees limited scale and spatial diversity. Mosaic addresses this by compositing 4 images into a 2×2 grid, forcing the model to detect objects at half scale and across varied anatomical contexts in a single forward pass.

**Design — cross-scan constraint:** The 3 mosaic partners are always drawn from 3 *different* scan IDs (different from the primary and from each other). This ensures each quadrant comes from a distinct acquisition, maximising context diversity and preventing near-duplicate slices from appearing in the same mosaic (which would reintroduce the redundancy the ScanAwareSampler eliminates).

**Implementation (`data/dataset.py`):**
- `__getitem__` now branches: with probability `mosaic_prob` calls `_load_mosaic(idx)`, otherwise `_load_single(idx)`. `mosaic_prob=0.0` is identical to prior behaviour.
- `_load_raw(idx)` → loads image + boxes at target_size, no augmentation.
- `_apply_augment(img, boxes, labels)` → applies albumentations transform (if configured) without `to_tensor`. Each mosaic quadrant is augmented independently.
- `_load_mosaic(idx)` → loads 4 images from 4 different scans, applies independent augmentation per quadrant, resizes each to 320×320, composites into a 640×640 canvas. Boxes scaled ×0.5, offset by quadrant origin, clipped, degenerate boxes (<4 px width/height) discarded. Falls back to single image if fewer than 4 scans exist in the split.
- `_load_single(idx)` → standard path, identical output to previous `__getitem__`.

**On realism:** Mosaic produces images that no ultrasound probe can generate (4 fan regions tiled). This is intentional — augmentations need only preserve label validity (bounding boxes remain geometrically correct), not physical plausibility. The tile boundaries are spatially random and uncorrelated with lesion positions, so the model learns to ignore them. The black fan borders appearing at internal tile edges similarly carry no label information.

**Integration:**
- `train.py`: added `--mosaic_prob` (float). Passed to `UltrasoundDataset(mosaic_prob=...)`. Val dataset always uses `mosaic_prob=0.0`.
- `train/config/default.yaml`: `mosaic_prob: 0.5` (half of training samples are mosaics).
- To disable: set `mosaic_prob: 0.0` in config or `--mosaic_prob 0`.

### 14. Per-Slice AP Metric (`utils/metrics.py`, `train/train.py`)

**Motivation:** The existing `mAP@50` is computed at the **object level** — each GT lesion box is an independent recall event, and all predicted boxes across the whole validation set are pooled into a single PR curve. This is the right metric when the goal is precise localization of every lesion.

However, a complementary question is: *does the model fire on the right slices at all?* A slice with 3 lesions where only 1 is detected contributes 1 TP and 2 FN to object-level mAP. At the slice level it is a correct detection regardless. This matters when:
- Comparing across datasets with different lesion densities (per-object scale differs, per-slice does not)
- Evaluating triage/screening performance (radiologist workflow: flag suspicious frames for review)
- Assessing sensitivity to annotation incompleteness — if some slices have partial GT, per-slice AP is more robust because it asks "was anything found" rather than "was every instance found"

**Implementation (`utils/metrics.py` — `Evaluator._per_slice_ap`):**
- For each slice: `score = max prediction confidence` (0.0 if no predictions)
- A slice is **positive** if it has ≥1 GT box; **negative** otherwise
- A positive slice is a **hit** if any predicted box has IoU ≥ 0.5 with any GT box
- Slices are sorted by score descending and a cumulative PR curve is built at the slice level
- AP is computed using standard 101-point interpolation (COCO convention)
- False alarms on negative (GT-free) slices penalize precision exactly as in the per-object case

**Integration:**
- `Evaluator.compute()` now returns `slice_ap_50` alongside all existing metrics
- `train.py`: `slice_ap_50` logged per epoch to JSON and printed as `SliceAP@50` in the epoch line

### 15. Full Training Reproducibility (`train/train.py`)

**Problem:** Training runs were non-deterministic across restarts — `set_seed()` seeded PyTorch, NumPy, and Python `random`, but three sources of non-determinism remained:
1. CUDA non-deterministic ops (scatter, index_add, upsample, etc.) were unrestricted.
2. Albumentations 2.x maintains its own internal RNGs (a NumPy `Generator` and a `random.Random`) that are separate from the global state seeded by `set_seed()`. DataLoader workers inherit process state at fork time but are not re-seeded by `seed_worker`.
3. `ScanAwareSampler` used a fixed internal seed (`seed=42`) regardless of the training seed, so the sampling sequence was identical across all runs rather than being governed by `--seed`.

**Fixes (`train/train.py`):**

- **CUDA determinism:** Added `torch.use_deterministic_algorithms(True, warn_only=True)` and `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` inside `set_seed()`. `warn_only=True` surfaces any remaining non-deterministic ops (e.g. from the model backbone) without crashing, allowing them to be addressed incrementally.
- **Albumentations RNG seeding:** Extended `seed_worker()` to call `dataset.transform.set_random_seed(worker_seed)` per worker. This seeds Albumentations' internal RNGs with the worker-specific seed (derived from PyTorch's initial seed) rather than a fixed global state.
- **`ScanAwareSampler` seed propagation:** `ScanAwareSampler` now accepts a `seed` argument (forwarded from `args.seed`). The sampler's internal RNG is initialised from this seed, so different `--seed` values produce different (but reproducible) sampling sequences.

**Result:** Two runs with the same `--seed` value now produce identical loss curves, metric sequences, and saved checkpoints.

### 16. Speckle-Specific Intensity Augmentations — MedianBlur + MultiplicativeNoise

**Motivation:** The previous augmentation config used `GaussianBlur` and `GaussNoise` as intensity augmentations. Both are poor models of ultrasound physics:

- **GaussianBlur** is a linear, isotropic low-pass filter. It blurs tissue boundaries and lesion edges indiscriminately. In clinical practice, ultrasound preprocessing uses speckle-reduction filtering (e.g. median filtering, anisotropic diffusion) that selectively smooths the interior of homogeneous regions while preserving edges. GaussianBlur does not replicate this behaviour.
- **GaussNoise** adds i.i.d. zero-mean additive noise to each pixel — matching electronic/quantisation noise. However, the dominant noise source in ultrasound images is **speckle**, which arises from coherent interference of backscattered ultrasound waves. Speckle intensity is proportional to local tissue backscatter: the noise is **multiplicative**, not additive. `GaussNoise` models the wrong physical process.

**Replacements (`train.py` flags, `best_hparams_pu_loss_speckle_copy_paste.yaml`):**

- **`MedianBlur`** (`blur_limit=5, p=0.3`): Non-linear, edge-preserving filter. The median filter replaces each pixel by the median of its neighbourhood, which eliminates isolated high/low pixels (speckle) without blurring transitions between tissue regions. This matches the effect of variable speckle-reduction preprocessing applied by different ultrasound machines and reconstruction settings.
- **`MultiplicativeNoise`** (`multiplier=[0.9, 1.1], p=0.4`): Each pixel is multiplied by an independent random scalar drawn from $U[0.9, 1.1]$. This directly models the multiplicative speckle process: $I_{\text{aug}} = I \times \eta$, $\eta \sim U[0.9, 1.1]$. The narrow multiplier range avoids extreme contrast changes while still introducing biologically plausible spatial intensity variation.

**Integration (`train.py`):**
- Added `--use_median_blur` / `--median_blur_limit` / `--median_blur_p` flags.
- Added `--use_multiplicative_noise` / `--multiplicative_noise_multiplier_min` / `--multiplicative_noise_multiplier_max` / `--multiplicative_noise_p` flags.
- The flags append the corresponding `albumentations` transforms after any YAML-defined augmentations in `aug_list`.

**Current status:** In progress — running experiments with `best_hparams_pu_loss_speckle_copy_paste.yaml` which uses MedianBlur + MultiplicativeNoise in place of GaussianBlur + GaussNoise. CLAHE and RandomBrightnessContrast are also removed in this config (consistent with HPO findings that CLAHE/RBC do not help with this scanner).

### 17. Tissue-Aware Copy-Paste Augmentation (`data/augmentations.py`)

**Motivation:** Labeled lesion instances are sparse (~10% label coverage). Standard copy-paste augmentation pastes lesion crops at arbitrary image locations, which can produce physically implausible composites (e.g. lesion pasted on the ultrasound machine border or inside another lesion). This can introduce noisy training signal. A tissue-aware variant instead constrains paste locations to regions with tissue statistics similar to the source lesion's background, ensuring the pasted crop is at least locally consistent with surrounding tissue.

**Algorithm (`TissueAwareCopyPaste.__call__`):**

1. **Valid-tissue mask:** Binarise the grayscale image at `tissue_threshold=10` to exclude the black machine border and background. Erode the mask by the crop bounding box size to guarantee every candidate centre places the full crop inside tissue.
2. **Source statistics:** For each GT bounding box (the lesion to copy), compute the mean and std of a circular neighbourhood of radius `radius=30 px` around the box centre. This characterises the tissue background surrounding the lesion.
3. **Candidate scoring:** Sample up to `n_candidates=50` positions from the eroded tissue mask. For each, compute the same neighbourhood statistics and score the candidate as $|\Delta\mu| + |\Delta\sigma|$. The candidate with the lowest score (closest tissue context to source) is selected.
4. **Threshold rejection:** If the best score exceeds `std_tol=25`, the paste is rejected — no sufficiently similar tissue region exists.
5. **Exclusion zones:** Before scoring, the valid mask is zeroed over (a) all existing GT bounding boxes (prevent pasting on top of a real annotation) and (b) a `±2·radius` region around the source lesion (prevent the scoring function from selecting back the origin neighbourhood).
6. **Paste and annotate:** The crop is pasted at the selected location and a new bounding box with the source class label is appended to the GT.
7. **Repeat:** Up to `max_pastes=2` new instances are created per image from shuffled GT pairs.

**Key design decisions:**
- **Shuffle source boxes** before iterating so the cap `max_pastes` does not always favour the first annotated box.
- **Per-source exclusion mask** (not a global one) so different source boxes can still overlap in their candidate regions — only each source's own neighbourhood is excluded.
- The paste does not blend edges (hard copy). This is intentional: blending would require alpha-matte estimation and is unnecessary since the tissue-match constraint already ensures local consistency.

**Integration:**
- `train.py`: reads `copy_paste` block from YAML config; instantiates `TissueAwareCopyPaste` if `enabled: true`; passes it to `UltrasoundDataset(copy_paste=...)`.
- `data/dataset.py`: `copy_paste` is applied in `_load_single` after `_load_raw` and before the Albumentations transform, so the pasted instances are subject to the same downstream augmentations.
- `train/config/default.yaml`: `copy_paste.enabled: false` (opt-in). Active in `best_hparams_pu_loss_speckle_copy_paste.yaml` with `p=0.4`, `max_pastes=2`.

**Current status:** In progress — running with the speckle + copy-paste combined config. Expected benefit: higher instance diversity per epoch especially for rare lesion sizes; model forced to detect lesions across varied tissue backgrounds.

### Next Steps
- Evaluate run results with MedianBlur + MultiplicativeNoise + TissueAwareCopyPaste.
- Compare against PU-Loss baseline (AP@50 = 0.557) to isolate contribution of speckle augmentations and copy-paste.
- If val loss continues diverging after epoch ~10, consider increasing `weight_decay` further or reducing `epochs` + relying on best-AP checkpoint.
- Pseudo-labeling on unlabeled data is a viable next step if AP plateaus — run best checkpoint on unlabeled images, keep high-confidence detections as new training labels, retrain.
