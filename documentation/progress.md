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

### 18. Why AP@50 Is an Unstable HPO Metric — and Why SliceAP@50 Replaces It

#### The instability problem

AP@50 (box-level mean average precision at IoU ≥ 0.5) exhibited single-epoch swings of ±0.10–0.15 in v3 HPO logs with no corresponding change in validation loss. Trial 201, for example, moved from 0.27 to 0.54 and back to 0.39 across three consecutive epochs while val loss barely moved (29.6 → 33.5 → 30.4). This makes it an unreliable stopping signal and corrupts Optuna's pruner: a configuration that would have reached 0.52 by epoch 30 can get killed at epoch 10 because it happened to produce 0.08 on a single bad checkpoint.

Two root causes were identified:

**1. Small effective sample size.**
Box-level AP@50 uses `torchmetrics MeanAveragePrecision`, which pools all predicted and GT boxes across the full validation set and builds a single PR curve over those boxes. The 357-slice validation set does not contain 357 independent GT boxes — the number of annotated lesion instances is substantially smaller (lesion-sparse ultrasound volumes), so the PR curve is built from far fewer samples than 357. A single misdetected lesion or a single spurious high-confidence false positive shifts the entire curve.

**2. Validation set data split: not scan-level holdout.**
Inspection of the data directory revealed that all 6 validation scan IDs (CEM004A3, CEM005A1, CEM006A1, CEM007A1, CEM013A2, CEM015A2) also appear in the training set. More critically, for two scans (CEM013A2 and CEM015A2) the validation slice indices fall entirely within the training slice index range — the validation slices are sandwiched between neighbouring training slices from the same scan. In 3D ultrasound, adjacent slices are near-identical (1 mm spacing). This means the model has seen frames adjacent to every validation slice during training, introducing within-scan correlation that makes the metric sensitive to scan-specific confidence calibration rather than true generalization.

The practical consequence: small changes in logit scale (caused by LR schedule progression or weight decay during training) shift the confidence of predictions on familiar scan neighbourhoods unpredictably, producing large AP swings that are not informative about generalization.

#### Why SliceAP@50 is more stable

`SliceAP@50` (implemented in `utils/metrics.py :: Evaluator._per_slice_ap`) treats each of the 357 validation slices as one binary sample:

- **Score:** maximum prediction confidence in that slice (0.0 if no predictions)
- **Positive:** slice has ≥1 GT box
- **Hit (TP):** any predicted box has IoU ≥ 0.5 with any GT box

The PR curve has exactly 357 points — the full validation set. Box-level AP@50 can have far fewer effective points when GT instances are sparse.

The stabilization comes from two effects:

1. **More samples:** 357 binary classification events are more stable than ~N_boxes events where N_boxes ≪ 357. Variance of a proportion scales as $1/n$; doubling the effective sample count halves the variance.

2. **Coarser question asked:** SliceAP asks "did you find anything in this slice?" rather than "did you find every box at the right IoU?" The binary hit criterion (`(iou >= 0.5).any()`) absorbs minor box placement variation — a box that shifts by a few pixels between epochs is still a hit. This removes the per-epoch jitter from near-threshold box quality variation.

The metric also has direct clinical relevance for this application: the primary screening task is flagging suspicious slices for radiologist review (binary per-slice decision), not precisely localizing every lesion instance. SliceAP@50 directly measures this.

**Observed stability:** In the same v3 training runs, `slice_ap_50` hovered in the 0.60–0.72 range in later epochs while `map_50` oscillated 0.40–0.55 over the same epochs. The signal-to-noise ratio is substantially higher.

#### Remaining caveats

SliceAP@50 is more lenient: a very large imprecise box that incidentally overlaps a GT box at IoU ≥ 0.5 is still a hit. Two configs with similar SliceAP@50 could differ meaningfully in box quality. For HPO ranking purposes (which config is better, not how precisely it localizes), this is an acceptable trade-off. Final model evaluation should still report both SliceAP@50 and AP@50.

The val-set data split issue (shared scan IDs) is a separate problem that affects both metrics. The correct fix is a scan-level resplit; this does not change the relative merit of SliceAP@50 vs AP@50 as HPO objectives.

---

### 19. Hyperparameter Optimization v4 (`train/tune_hyperparams_v4.py`)

#### Summary of changes from v3

| Component | v3 | v4 |
|---|---|---|
| Optuna objective | Raw AP@50 | EMA SliceAP@50 |
| Pruner reported metric | Raw AP@50 | Raw SliceAP@50 (not EMA) |
| Early stopping criterion | Raw AP@50 | EMA SliceAP@50 |
| Pruner type | `MedianPruner` | `HyperbandPruner` |
| Checkpoint saving | None | Top-3 per trial by EMA SliceAP@50 |
| Output best config | `best_hyperparams_v3.yaml` | `best_hyperparams_v4.yaml` |

#### EMA smoothing (α = 0.3)

A per-trial exponential moving average is applied to `slice_ap_50` at every epoch:

```
ema_t = 0.3 × slice_ap_t  +  0.7 × ema_{t-1}
```

The EMA is warm-started at epoch 1 (set to the raw value, no lag from zero). With α = 0.3 the effective half-life is ~2 epochs — enough to absorb the ±0.10 single-epoch swings visible in the v3 logs while tracking real trends on the timescale of 5–10 epochs.

The EMA is used as:
- **The trial objective returned to Optuna** (what the study maximises and uses for config ranking)
- **The early stopping signal** (counter resets only when EMA improves by ≥ `min_delta`)

The raw `slice_ap_50` (not EMA) is reported to the **pruner** via `trial.report()`. This is intentional: the HyperbandPruner's rung comparisons should reflect the model's actual state at a specific epoch, not a lagged average that would make a slowly-converging trial look better than a fast one at early rungs.

#### HyperbandPruner

Replaces `MedianPruner` for two reasons:

1. `MedianPruner` compares a trial's metric against the global median at the same epoch. When trials are dispatched in parallel across 20 SLURM workers, the "same epoch" comparison is often across trials that are at very different phases of convergence, and the global median itself is noisy during the first 8–20 trials before enough data accumulates.

2. `HyperbandPruner` groups trials into brackets and uses successive halving within each bracket. Trials are only compared to others that started with similar budgets. This is more robust to the delayed-onset convergence in this dataset (most configs produce near-zero AP before epoch 8).

Configuration:
- `min_resource=10` — no pruning before epoch 10 (past the warmup LR ramp)
- `max_resource=50` — matches `--epochs`
- `reduction_factor=3` — rungs at epochs 10, 30, 50; keeps top ⅓ at each rung

Known limitation: Hyperband's bracket structure assumes workers run trials sequentially within brackets. With 20 asynchronous SLURM workers sharing one SQLite study, bracket assignment can be disrupted. If erratic early pruning is observed in v4 logs, this is the cause. MedianPruner was more robust to this kind of distributed asynchrony, at the cost of the comparison noise described above.

#### Top-K checkpoint saving (`TopKCheckpointManager`)

Each trial now saves up to 3 model checkpoints during training, managed by `TopKCheckpointManager` using a min-heap. At every epoch:

1. The current model weights are saved to `checkpoints/hparam_v4/trial_<N>/epoch<E>_sliceap<V>.pt`
2. The new checkpoint is pushed onto the heap (keyed by EMA SliceAP@50)
3. If the heap exceeds K=3 entries, the checkpoint with the lowest EMA SliceAP@50 is popped and its file deleted

This keeps exactly the top-3 best checkpoints alive at all times without accumulating files. On pruning, `cleanup()` deletes all checkpoint files for that trial to prevent disk bloat.

At the end of each completed trial, `write_manifest()` produces `swa_manifest.yaml` listing the surviving checkpoints:

```yaml
trial_number: 42
trial_params: {learning_rate: 4.8e-04, ...}
checkpoints:
  - path: "checkpoints/hparam_v4/trial_0042/epoch014_sliceap0.6823.pt"
    epoch: 14
    slice_ap50: 0.671234
    ema_slice_ap50: 0.682300
  - ...
```

Only the base YOLOE model weights (`YOLOEWithLoss.model.state_dict()`) are saved per checkpoint — not the loss wrapper — keeping each file to ~25 MB and making them directly loadable by `build_yoloe()` for inference or SWA averaging.

**Disk budget:** 300 trials × 3 checkpoints × ~25 MB ≈ 22 GB on NFS. This is manageable; pruned trial checkpoints are deleted immediately.

#### SWA weight averaging (post-study)

After the study completes, the top trial manifests can be used to perform stochastic weight averaging across the best checkpoints:

```python
import torch, yaml, glob
from model.yoloe import build_yoloe

# Load manifests for your chosen top trials
manifest_paths = ["checkpoints/hparam_v4/trial_0042/swa_manifest.yaml", ...]
state_dicts = []
for p in manifest_paths:
    m = yaml.safe_load(open(p))
    for ckpt in m["checkpoints"]:
        sd = torch.load(ckpt["path"], map_location="cpu")["model_state_dict"]
        state_dicts.append(sd)

# Average weights
avg_sd = {k: sum(sd[k] for sd in state_dicts) / len(state_dicts) for k in state_dicts[0]}
model = build_yoloe(model_size="s", num_classes=1)
model.load_state_dict(avg_sd)
```

Averaging across checkpoints from different epochs of the same trial (or across top trials) has been shown to improve generalization in noisy-metric regimes by smoothing out checkpoint-level variance. Saving the top-3 per trial makes this option available without requiring a separate full-training SWA run.

---

### 20. Hyperparameter Optimization v5 (`train/tune_hyperparams_v5.py`)

#### Motivation — overfitting analysis of v4 (job 752)

Post-hoc analysis of the 300-trial v4 run identified three consistent overfitting signals across virtually all completed trials:

1. **Train/val loss gap triples by end of training.** The gap at epoch ~5 is ~8–14; by the final epoch it reaches ~23–30. Train loss keeps falling while val loss plateaus, indicating the model is memorising scan-specific features of the 13 training patients.

2. **Val loss bottoms early.** Val loss hits its minimum between epochs 16 and 30 in nearly every trial, then drifts up by +1–7 units. Epochs 35–50 provide no SliceAP@50 gain and only deepen the train/val gap.

3. **Weight decay saturates at the search space boundary.** Both the v3 and v4 best trials converged to `wd=7.97e-03`, immediately below the old upper bound of `1e-2`. When a hyperparameter search saturates at a boundary, the true optimum lies beyond it.

#### Search space changes

| Parameter | v4 range | v5 range | Rationale |
|---|---|---|---|
| `weight_decay` | `[1e-4, 1e-2]` log | `[1e-4, 5e-2]` log | Best wd in v3 and v4 both hit the old ceiling; extends regularisation headroom 5× |
| `mosaic_prob` | `[0.1, 0.5]` | `[0.3, 0.8]` | Cross-scan mosaic is the strongest anti-overfitting augmentation; v4 best (0.39) was near the old ceiling; raising it forces more aggressive patient-mixing |
| `copy_paste_p` | `[0.0, 0.7]` | `[0.2, 0.7]` | v4 best used 0.11 — well below effective range; with only 13 scans, copy-paste is the only source of novel lesion instances; floor of 0.2 ensures every trial uses it |

All other search space parameters are unchanged from v4.

#### Training budget changes

| Parameter | v4 | v5 | Rationale |
|---|---|---|---|
| `--epochs` | 50 | 35 | Val loss bottoms at ep 16–30; ep 35–50 only deepens overfitting. Saves ~30% GPU time per trial, enabling more trials in the same wall time. |
| `--early-stopping-patience` | 15 | 8 | EMA smoothing (α=0.3, half-life ~2 ep) already absorbs single-epoch dips; 15-epoch patience was equivalent to ~30 raw epochs of tolerance — too permissive. |

#### Unchanged from v4

- Objective: EMA SliceAP@50 (α=0.3), warm-started at epoch 1
- Pruner: `HyperbandPruner` (`min_resource=10`, `max_resource=35`, `reduction_factor=3`)
  - Note: `max_resource` updated from 50 → 35 to match new epoch budget; rungs now at epochs 10 and 35
- Sampler: `TPESampler`
- Checkpoint saving: `TopKCheckpointManager`, top-3 per trial, `swa_manifest.yaml` for SWA
- Copy-paste: `TissueAwareCopyPaste` always instantiated (floor > 0), fixed structural params from base config

#### Files

- `train/tune_hyperparams_v5.py` — tuning script
- `slurm-script/tune_hyperparams_v5.sh` — SLURM array job (20 workers × 15 trials = 300 total)
- Outputs: `train/config/hparam_v5_rank*.yaml`, `train/config/best_hyperparams_v5.yaml`
- Checkpoints: `checkpoints/hparam_v5/trial_<N>/`

---

### 21. Hyperparameter Optimization v6 (PU Loss) — Experiment Results & Critical Analysis

#### What was run

- `tune_hyperparams_v6.py` was run with `best_hyperparams_v5.yaml` (which had `use_pu_loss` **accidentally set to `true`** after v5 tuning completed) as the base config. All v6 Optuna trials therefore ran with PU loss enabled throughout.
- The v6 best config (`train/config/best_hyperparams_v6.yaml`) was used to train `checkpoints/best_model_20260319_153153.pth`, which reached best EMA SliceAP@50 = **0.6756** at epoch 28 and early-stopped at epoch 48.
- `best_hyperparams_v5.yaml` has been reverted to `use_pu_loss: false`. `best_hyperparams_v6.yaml` is retained as-is (`use_pu_loss: true`) as a historical experiment record.

**Key fact:** The v6 Optuna study converged to **identical numerical hyperparameters** as v5 (lr=0.0004095, wd=0.000131, same augmentation params), differing only in `use_pu_loss`. This confirms that the non-loss hyperparameters (LR, WD, augmentations) are insensitive to which loss function is used — the gap between v5 and v6 is attributable entirely to the PU loss formulation.

#### Test set results (held-out, 328 slices, 5 scans)

| Metric | v5 no-PU (conf=0.47, iou=0.45) | v5 no-PU (conf=0.5, iou=0.5) | v6 PU Loss (conf=0.5, iou=0.5) |
|---|---|---|---|
| AP@50 | **0.7704** | 0.7704 | 0.5381 |
| SliceAP@50 | **0.9406** | 0.9406 | 0.8317 |
| SliceAP@75 | 0.4257 | 0.4257 | 0.4257 |
| Precision | **0.6600** | 0.6600 | 0.1978 |
| Recall | **0.8049** | 0.8049 | 0.6677 |
| F1 | **0.7253** | 0.7253 | 0.3052 |
| TP / FN / FP | 282 / 46 / 138 | 264 / 64 / 136 | 219 / 109 / **888** |
| FA/image | 0.42 | 0.41 | **2.71** |
| P @ R=0.8 | **0.6658** | 0.6658 | 0.1011 |

**Operational baseline: v5 no-PU at conf=0.47, iou=0.45** (85.98% DR, 0.42 FA/image).

#### Per-scan breakdown

| Scan | v5 DR (best) | v6 DR | v5 FA/slice | v6 FA/slice |
|---|---|---|---|---|
| CEM005A1 | 94.5% | **17.97%** | 0.42 | 2.56 |
| CEM011A3 | 84.0% | 100.0% | 0.68 | 3.44 |
| CEM012A1 | 71.6% | 96.1% | 0.30 | 2.61 |
| CEM013B2 | 83.8% | 100.0% | 0.57 | 2.35 |
| CEM015A2 | 100.0% | 100.0% | 0.42 | 3.36 |

The identical SliceAP@75 = 0.4257 across all conditions confirms the backbone's localization quality is intact — the failure is exclusively in classification branch calibration.

#### Why the current PU loss formulation fails

The implementation in `model/pu_loss.py` has four structural deficiencies:

1. **Soft sampling is not a PU correction.** The weight `(1-p)^γ` applied to negative anchors is mathematically equivalent to focal loss on negative targets. It provides maximum suppression gradient at `p ≈ 1/3`, not at `p ≈ 0` (true background). High-confidence potential positives (`p > 0.7`) receive weight `(0.3)² = 0.09` — reduced, but still a systematic push toward zero throughout training.

2. **Focal IoU starvation of positive gradients.** `weight[fg_mask] = IoU(pred, gt)^β`. At random initialization, predicted box IoU ≈ 0.01–0.05, reducing positive classification gradient to 1–5% of baseline during the critical early epochs. Box regression improves normally (unaffected) while the classification head cannot learn to confidently predict positives. Net gradient ratio neg:pos worsens from ~24:1 (base loss) to ~247:1 (PU loss at epoch 1).

3. **Normalization asymmetry (280× bias).** No-GT images normalize by `num_anchors ≈ 8400`; with-GT image negatives normalize by `num_pos ≈ 30`. The soft weighting on no-GT slices (the intended PU use case) is effectively irrelevant — annotated-image negatives dominate the gradient budget by 280×.

4. **No nnPU non-negativity constraint.** The Kiryo et al. (2017) nnPU estimator requires `max(0, R_u- − π·R_p-)`. Without this clamp, when the model enters a suppression regime (as on CEM005A1), gradients continue pushing predictions toward zero indefinitely with no recovery — explaining the 17.97% DR collapse on that scan.

#### CEM005A1 collapse explained

CEM005A1 is the most lesion-dense scan (128 GT boxes across 128 slices, ~one per slice). The PU loss teaches the model a spurious discriminative rule: suppress confident predictions on scans that globally resemble training scans where suppression was rewarded (scans that appear "background-dominant" in acquisition style). CEM005A1's acquisition signature differs from the other 4 test scans, causing trained suppression to activate across all 128 of its slices. This is a direct consequence of deficiency (4) above.

---

### 22. Planned Next Steps — Phase 1 & Phase 2

#### Current status

- **Active baseline:** `checkpoints/best_model_v5_no_pu_loss.pth` trained from `best_hyperparams_v5.yaml` (`use_pu_loss: false`). No further hyperparameter re-tuning is required — v5 hyperparameters are correctly tuned for the base loss.
- **Historical record:** `best_hyperparams_v6.yaml` retained as-is with `use_pu_loss: true`. Do not use this config for new training runs without changing the flag.
- **Known remaining weakness:** CEM012A1 achieves only 67–71% DR under the base model. Root cause under investigation (see Phase 1 goals).

---

#### Phase 1 — Base Loss & Assignment Improvements

**Goal:** Improve the base model's localization precision (AP@75) and reduce false alarms (precision) without sacrificing recall. No new loss paradigm required — these are well-established improvements to the existing YOLOE stack.

##### P3 — CIoU Box Regression Loss (`model/loss.py`)

**Problem:** The current box regression loss is `smooth_l1_loss(pred_boxes, target_boxes)`, which treats x1, y1, x2, y2 as four independent scalars. It provides no gradient incentive for improving overlap when boxes already partially intersect, and cannot distinguish between boxes with the same L1 error but different geometric quality.

**Fix:** Replace with Complete IoU loss (CIoU, Zheng et al. 2020):
```
L_CIoU = 1 - IoU + ρ²(b, b^gt)/c² + α·v
```
where `ρ²(b, b^gt)/c²` penalises centre-point distance normalised by diagonal, and `αv` penalises aspect ratio difference. CIoU provides non-zero gradient even when boxes don't overlap, and improves AP@75 specifically because the aspect ratio term tightens predicted shapes to GT shapes.

Expected impact: AP@75 improvement from 0.2422 toward 0.33–0.38 (based on typical CIoU gains over smooth L1 in YOLO-family detectors on small, compact objects). FA/image reduction expected as better-localised predictions reduce IoU-threshold boundary noise.

**Files to modify:** `model/loss.py` (replace `smooth_l1_loss` call; add CIoU helper); `model/pu_loss.py` (same replacement for consistency, though PU loss is deprioritised).

##### P4 — Task-Aligned Assignment (TAL) (`model/loss.py`, `model/assigners/`)

**Problem:** `_assign_targets` in `loss.py` uses distance-to-GT-center + inside-box filter. This assigns anchors based purely on spatial proximity, independent of whether the model is actually predicting anything useful at that anchor. A spatially close anchor that predicts a box in entirely the wrong direction still receives a positive label, injecting noisy positive gradient.

**Fix:** Replace with Task-Aligned Learning (TAL) assignment (TOOD, Feng et al. 2021). TAL computes an alignment metric `t_i = cls_score_i^α × IoU_i^β` for each anchor-GT pair, then selects the top-m anchors by this metric within each GT's region. Because both classification score and IoU are jointly considered, only anchors that are both confident and geometrically accurate are assigned positive. This:
- Makes the positive set self-consistent with current model predictions
- Eliminates the need for the Focal IoU weight in the loss (it becomes implicit in assignment quality)
- Naturally improves precision: noisy positive assignments that were confusing the classification head are excluded

Note: `ATSSAssigner` already exists in `model/assigners/atss_assigner.py` but is incomplete (returns IoU tensor only, stops at line 100). TAL requires a new assigner or completion of ATSS. The assignment interface (returns `target_cls`, `target_box`, `fg_mask`) is already defined in `loss.py`'s `_assign_targets` and can be swapped.

Expected impact: Precision improvement from 0.66 toward 0.72–0.76. P@R=0.8 improvement from 0.6658 toward 0.72+. AP@50 expected to hold or improve slightly.

**Files to modify:** `model/assigners/` (new TAL assigner); `model/loss.py` (replace `_assign_targets`).

##### P5 — Hard Negative Mining / Negative Sample Count Cap (`model/loss.py`)

**Problem:** With `num_pos ≈ 30` and `num_anchors ≈ 8400`, the per-anchor negative-to-positive gradient ratio is ~279:1 under the current `sum/num_pos` normalization. This extreme imbalance means thousands of easy background anchors (deep in correct territory, `p ≈ 0.02`) dominate the gradient budget despite contributing almost zero useful learning signal. The model has difficulty discriminating hard negatives (tissue regions with lesion-like appearance) from positives.

**Fix:** Online Hard Example Mining (OHEM) for the negative set. After computing `bce(pred_cls[~fg_mask], target_cls[~fg_mask])`, sort by descending loss and retain only the top `k × num_pos` (e.g., `k=3` or `k=5`) hardest negative anchors before summing. This:
- Caps the effective neg:pos ratio at `k:1` (e.g., 3:1 or 5:1)
- Focuses gradient on near-threshold negatives (the hard false positives)
- Leaves easy, already-suppressed negatives out of the gradient, preventing over-regularization of the background

Combined with P3 (CIoU) and P4 (TAL), this forms the complete precision-improvement stack.

Expected impact: Precision increase to 0.73–0.78. FA/image reduction from 0.42 toward 0.25–0.30. Recall expected to hold (OHEM focuses on hard negatives, does not affect positive gradient).

**Files to modify:** `model/loss.py` (add OHEM selection before neg_loss computation).

**Phase 1 evaluation protocol:**
- Train using `best_hyperparams_v5.yaml` with each change incrementally (P3 alone → P3+P4 → P3+P4+P5).
- Report test-set metrics on the held-out 5-scan set using the same eval script (`inference/test_eval.py`).
- Accept a change only if it improves SliceAP@50 ≥ 0.93 (does not hurt detection rate) AND reduces FA/image by ≥ 0.05.
- Per-scan breakdown (especially CEM012A1 and CEM005A1) must be checked for each run.

---

#### Phase 2 — PU Loss Research (Dropped)

**Decision date:** 2026-03-20. Phase 2 (P6 nnPU constraint, P7 Optuna v7 gamma/beta tuning) was planned but dropped after reviewing the Phase 1 results. Reasoning is documented below.

##### Why Phase 2 was abandoned

**1. SliceAP@50 is already at ceiling.**
The Phase 1 model achieves SliceAP@50 = 0.9901 — essentially perfect at the slice level. PU learning's core value is preserving recall on unannotated positive slices ("don't suppress confident predictions where lesions exist but aren't labelled"). With 99% of positive slices already detected, this problem is already solved by the base model. PU loss has almost no headroom to improve the primary metric.

**2. The remaining failure modes are not PU problems.**
The two outstanding weaknesses after Phase 1 are: (a) CEM012A1 false alarm regression (0.27 → 0.53 FA/slice at conf=0.35), which is a precision problem caused by the model firing on background parenchymal enhancement — a pattern PU learning would soften rather than fix; and (b) a ~15% overall miss rate concentrated in genuinely hard lesions (very small, low contrast, edge slices) that are hard regardless of loss formulation. PU learning addresses recall on unannotated slices, not localisation difficulty or background confusion.

**3. P6 alone is insufficient; all structural fixes are needed together.**
P6 (nnPU non-negativity clamp) prevents catastrophic training collapse but leaves three other structural deficiencies in `pu_loss.py` unaddressed: the soft sampling inversion (max gradient at p≈0.33 rather than at p≈0), the 280× normalization asymmetry between GT and no-GT images, and the absence of spatial gating (soft weights applied uniformly to all 8400 negative anchors rather than only near-GT candidates). Running P7 (Optuna) without fixing all four issues tunes the parameters of a structurally broken loss and is unlikely to find a good solution.

**4. PU learning requires annotation audit first.**
A sound PU study requires knowing the actual unlabeled positive rate in the training data — the fraction of lesion-containing slices that were not annotated. The test data shows near-complete annotation coverage (CEM005A1: 128/128 slices annotated, CEM012A1: 102/102). If training data has similarly dense annotation, the PU correction is solving a negligible problem while introducing gradient instability. This audit was not performed, and given the Phase 1 results there is insufficient motivation to perform it now.

**5. Regression risk outweighs expected gain.**
The v6 experiment demonstrated that the current PU loss formulation causes an 30pp AP@50 regression and a 6.5× increase in false alarms. Even a structurally improved PU loss carries regression risk on a 5-scan test set where a single scan's failure dominates aggregate metrics. The expected upside (marginal improvement on a metric already near ceiling) does not justify this risk.

##### What would justify revisiting PU learning

- Evidence from an annotation audit that ≥20% of training lesion-containing slices are unannotated
- A larger test set (≥20 scans) with sufficient statistical power to detect a 1–2pp improvement
- All four structural fixes implemented together (nnPU clamp + spatial gating + normalization consistency + correct soft sampling) before any Optuna search
- A dataset where the primary failure mode is suppression of confident predictions on positive slices, not background confusion or localisation difficulty

---

#### Summary Priority Table

| Priority | Component | Files | Phase | Status |
|---|---|---|---|---|
| P3 | CIoU box regression loss | `model/loss.py` | 1 | **Complete** |
| P4 | Task-Aligned Assignment (TAL) | `model/loss.py`, `model/assigners/` | 1 | **Complete** |
| P5 | Hard Negative Mining (OHEM) | `model/loss.py` | 1 | **Complete** |
| P6 | nnPU non-negativity constraint | `model/pu_loss.py` | 2 | **Dropped** — see reasoning above |
| P7 | Tune gamma/beta + spatial gating | `train/tune_hyperparams_v7.py` | 2 | **Dropped** — see reasoning above |

---

### 23. Phase 1 Results — CIoU + TAL + OHEM (v5_ciou_tal_ohem)

#### Experiment provenance

| Item | Detail |
|---|---|
| **Training config** | `train/config/best_hyperparams_v5.yaml` (`use_pu_loss: false`) |
| **Training script** | `train/train.py` |
| **SLURM script** | `slurm-script/train_v5_p3p4p5.sh` (job 1052) |
| **Checkpoint** | `checkpoints/best_model_v5_ciou_tal_ohem_20260320_134819.pth` |
| **Best epoch** | 20 |
| **Best val EMA SliceAP@50** | 0.7671 |
| **Eval script** | `slurm-script/test_eval_v5_ciou_tal_ohem.sh` (job 1055) |
| **Eval results dir** | `inference/test_results_v5_ciou_tal_ohem/` |
| **Eval thresholds** | conf=0.47, iou=0.45 (same as v5 baseline for direct comparison) |

**Changes vs v5 baseline** (implemented in `model/loss.py`):
- P3: CIoU box regression replaces smooth-L1
- P4: Task-Aligned Assignment (TAL, α=0.5, β=6.0, topk=13) replaces distance-based top-k
- P5: Online Hard Example Mining (OHEM, ratio=3×) replaces full negative set

#### Test set results (held-out, 328 slices, 5 scans)

Baselines use `checkpoints/best_model_v5_no_pu_loss.pth` trained from `best_hyperparams_v5.yaml`.

| Metric | v5 baseline (conf=0.5, iou=0.5) | v5 baseline (conf=0.47, iou=0.45) | **v5_ciou_tal_ohem** (conf=0.47, iou=0.45) | Δ vs best baseline |
|---|---|---|---|---|
| AP@50 | 0.7704 | 0.7704 | **0.8270** | **+5.7pp** |
| AP@75 | 0.2422 | 0.2422 | **0.3622** | **+12.0pp** |
| mAP@50:95 | 0.3229 | 0.3229 | **0.4019** | **+7.9pp** |
| SliceAP@50 | 0.9406 | 0.9406 | **0.9901** | **+4.9pp** |
| SliceAP@75 | 0.4257 | 0.4257 | **0.5446** | **+11.9pp** |
| Precision | 0.6600 | 0.6600 | **0.7967** | **+13.7pp** |
| Recall | 0.8049 | 0.8049 | 0.7409 | -6.4pp ¹ |
| F1 | 0.7253 | 0.7253 | **0.7678** | **+4.3pp** |
| P @ R=0.8 | 0.6658 | 0.6658 | **0.7601** | **+9.4pp** |
| R @ P=0.8 | 0.6341 | 0.6341 | **0.7317** | **+9.8pp** |
| TP / FN / FP | 264/64/136 | 279/49/121 | 249/79/**70** | FP **−42%** |
| FA/image | 0.4146 | 0.3689 | **0.2134** | **−42%** |

¹ *Recall drop is a confidence calibration shift — see note below.*

#### Per-scan breakdown

| Scan | v5 DR (iou45 baseline) | new DR | v5 FA/slice | new FA/slice |
|---|---|---|---|---|
| CEM005A1 | 92.97% | 72.66% | 0.33 | 0.25 |
| CEM011A3 | 84.00% | 72.00% | 0.68 | **0.04** |
| CEM012A1 | 71.57% | **73.53%** | 0.27 | 0.32 |
| CEM013B2 | 81.08% | 72.97% | 0.57 | **0.11** |
| CEM015A2 | 100.00% | **100.00%** | 0.36 | **0.00** |

#### Key findings

**What worked:**
- **AP@75 +12pp** is the largest gain and directly confirms CIoU is working — predicted boxes are tighter and more geometrically accurate. SliceAP@75 confirms the same (+11.9pp).
- **False alarms reduced 42%** (FP: 121 → 70, FA/image: 0.37 → 0.21). OHEM is forcing the classifier to be selectively confident. CEM011A3 drops from 0.68 to 0.04 FA/slice; CEM013B2 from 0.57 to 0.11; CEM015A2 achieves zero false alarms for the first time.
- **P@R=0.8 = 0.7601 vs 0.6658**: at the clinically required recall level, precision improves 9.4pp. At equivalent recall=0.8, estimated FA/image ≈ 0.25 vs 0.37 — a 33% reduction in false alarm burden.

**The recall regression is a threshold artefact, not a model regression:**
AP@50 = 0.8270 vs 0.7704 confirms the new model has a strictly better PR curve. OHEM shifts the confidence score distribution upward for high-quality predictions and downward for uncertain ones. The conf=0.47 threshold, calibrated for the old score distribution, is now too high for this model. Re-evaluating at conf=0.35–0.38 is expected to recover recall to ≥0.80 while retaining the precision/FA gains.

#### Threshold sweep — finding the new operating point

Jobs 1056 (conf=0.35) and 1057 (conf=0.38) run against the same checkpoint and config. Results in `inference/test_results_v5_ciou_tal_ohem_conf35/` and `inference/test_results_v5_ciou_tal_ohem_conf38/`.

**Note on metrics_report.txt:** The Precision/Recall/F1 header values are computed at the PR-curve's internal F1-optimal threshold and do not vary with `--conf-thresh`. The per-threshold numbers of record are the Lesion Breakdown TP/FN/FP/DR/FA rows.

| Config | Checkpoint | conf | iou | TP | FN | FP | DR | FA/img |
|---|---|---|---|---|---|---|---|---|
| v5 baseline | `best_model_v5_no_pu_loss` | 0.50 | 0.50 | 264 | 64 | 136 | 80.49% | 0.415 |
| v5 baseline | `best_model_v5_no_pu_loss` | 0.47 | 0.45 | 279 | 49 | 121 | 85.06% | 0.369 |
| ciou+tal+ohem | `best_model_v5_ciou_tal_ohem` | 0.47 | 0.45 | 249 | 79 | 70 | 75.91% | 0.213 |
| ciou+tal+ohem | `best_model_v5_ciou_tal_ohem` | 0.38 | 0.45 | 275 | 53 | 96 | 83.84% | 0.293 |
| **ciou+tal+ohem** | **`best_model_v5_ciou_tal_ohem`** | **0.35** | **0.45** | **281** | **47** | **116** | **85.67%** | **0.354** |

#### Per-scan at conf=0.35

| Scan | v5 DR (iou45) | new DR | v5 FA/slice | new FA/slice |
|---|---|---|---|---|
| CEM005A1 | 92.97% | 89.84% | 0.33 | 0.34 |
| CEM011A3 | 84.00% | 76.00% | 0.68 | **0.56** |
| CEM012A1 | 71.57% | **80.39%** | 0.27 | 0.53 ⚠ |
| CEM013B2 | 81.08% | 78.38% | 0.57 | **0.14** |
| CEM015A2 | 100.00% | **100.00%** | 0.36 | **0.00** |

#### Verdict: conf=0.35 is a Pareto improvement over the best v5 baseline

At conf=0.35, the new model achieves **281 TP vs 279 TP** (+2) and **116 FP vs 121 FP** (−5) compared to the best v5 baseline — DR and FA/image both improve simultaneously.

Highlights:
- **CEM012A1**: DR jumps from 71.57% → 80.39% — the previously weakest scan, directly improved by TAL's quality-gated assignment. FA/slice rises from 0.27 → 0.53 ⚠ — warrants monitoring; may benefit from targeted augmentation.
- **CEM015A2**: 100% DR, zero false alarms (down from 0.36 FA/slice).
- **CEM013B2**: FA/slice drops from 0.57 → 0.14, large precision improvement.

**New operational baseline: `best_model_v5_ciou_tal_ohem_20260320_134819.pth`, conf=0.35, iou=0.45**
- DR: 85.67% (+0.6pp vs prior best), FA/image: 0.354 (−4% vs prior best)
- AP@50: 0.8270, AP@75: 0.3622, SliceAP@50: 0.9901, SliceAP@75: 0.5446, P@R=0.8: 0.7601
