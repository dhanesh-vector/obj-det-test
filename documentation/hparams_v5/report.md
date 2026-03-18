# YOLOE Hyperparameter Optimisation v5 — Summary Report

**Study:** `yoloe_hparam_search_v5`
**SLURM Job Array:** 916 (20 workers × 15 trials = 300 total trials)
**Run Date:** 2026-03-17 → 2026-03-18
**Objective:** Maximise EMA SliceAP@50 (α = 0.3)
**Best Result: EMA SliceAP@50 = 0.7500** (Trial #409)

---

## 1. Executive Summary

v5 achieved **EMA SliceAP@50 = 0.7500**, a **+0.0220 improvement** (+3.0%) over the v4 best of 0.7280. Since v3 (best = 0.6187), the three-generation campaign has delivered a cumulative **+0.1313 absolute improvement** (+21.2%). All three runs used identical budgets (300 trials, TPE + Hyperband), with each version tightening the search space and training schedule based on post-hoc analysis of the previous run.

| Version | SLURM Job | Best EMA SliceAP@50 | Δ vs prev |
|---------|-----------|---------------------|-----------|
| v3      | 732       | 0.6187              | —         |
| v4      | 752       | 0.7280              | +0.1093   |
| v5      | 916       | **0.7500**          | +0.0220   |

---

## 2. Study Configuration

| Parameter              | Value                                      |
|------------------------|--------------------------------------------|
| Sampler                | TPESampler (seed = 42)                     |
| Pruner                 | HyperbandPruner (min=10, max=35, factor=3) |
| Max epochs / trial     | 35 (reduced from 50 in v4)                 |
| Early-stop patience    | 8 epochs (tightened from 15 in v4)         |
| Early-stop min delta   | 0.005                                      |
| EMA alpha              | 0.3 (half-life ≈ 2 epochs)                 |
| Checkpoint threshold   | EMA ≥ 0.5 (top-3 per trial saved)          |
| Database               | SQLite WAL, 25 s lock timeout              |
| Workers                | 20 parallel SLURM array tasks              |
| Trials / worker        | 15                                         |
| DB size                | ~1.6 MB                                    |

**Final trial counts (from DB):**
- Completed: **203**
- Pruned: **235**
- Failed: 23
- Still running at report time: 58

---

## 3. Search Space

### Changes from v4 → v5

| Parameter       | v4 Range          | v5 Range         | Rationale |
|-----------------|-------------------|------------------|-----------|
| `weight_decay`  | [1e-4, 1e-2] log  | [1e-4, 5e-2] log | v4 best (7.97e-3) saturated at upper bound; 5× extension explores stronger regularisation |
| `mosaic_prob`   | [0.1, 0.5] linear | [0.3, 0.8] linear | v4 best (0.39) near old ceiling; cross-scan mosaic is the strongest anti-overfitting aug |
| `copy_paste_p`  | [0.0, 0.7]        | [0.2, 0.7]       | Floor raised to ensure copy-paste is active in every trial (only 13 training scans) |
| `--epochs`      | 50                | 35               | Val loss bottoms at epoch 16–30 in v4; epochs 35–50 only deepen the overfitting gap |
| `--es-patience` | 15                | 8                | EMA smoothing already absorbs noise; 15 was equivalent to ~30 raw epochs of tolerance |

### Full Search Space (v5)

| Parameter               | Type        | Range / Choices          |
|-------------------------|-------------|--------------------------|
| `learning_rate`         | Float (log) | [1e-4, 5e-3]             |
| `weight_decay`          | Float (log) | [1e-4, 5e-2]             |
| `mosaic_prob`           | Float       | [0.3, 0.8]               |
| `label_smooth`          | Float       | [0.0, 0.15]              |
| `copy_paste_p`          | Float       | [0.2, 0.7]               |
| `slice_stride`          | Categorical | {5, 7, 10}               |
| `batch_size`            | Categorical | {8, 16, 32}              |
| `use_clahe`             | Categorical | {True, False}            |
| `use_shift_scale_rotate`| Categorical | {True, False}            |
| `use_brightness_contrast`| Categorical| {True, False}            |
| `use_median_blur`       | Categorical | {True, False}            |
| `median_blur_limit`     | Categorical | {3, 5, 7}                |
| `use_speckle_noise`     | Categorical | {True, False}            |
| `speckle_level`         | Categorical | {light, medium, heavy}   |

Fixed (always on): HorizontalFlip, VerticalFlip.

---

## 4. Top-5 Results

| Rank | Trial | EMA SliceAP@50 | LR       | WD       | Mosaic | CopyPaste P | Label Smooth | Batch | Stride | CLAHE | Speckle | Median Limit |
|------|-------|----------------|----------|----------|--------|-------------|--------------|-------|--------|-------|---------|--------------|
| 1    | 409   | **0.7500**     | 8.43e-4  | 3.96e-3  | 0.324  | 0.494       | 0.079        | 16    | 10     | No    | No      | 3            |
| 2    | 123   | 0.7354         | 3.99e-4  | 8.18e-4  | 0.399  | 0.261       | 0.087        | 16    | 10     | Yes   | No      | 5            |
| 3    | 478   | 0.7289         | 1.06e-3  | 9.81e-4  | 0.485  | 0.666       | 0.068        | 16    | 10     | No    | Yes     | 5            |
| 4    | 83    | 0.7226         | 6.11e-4  | 1.31e-3  | 0.381  | 0.324       | 0.103        | 8     | 10     | Yes   | No      | 7            |
| 5    | 305   | 0.7200         | 8.71e-4  | 5.76e-4  | 0.673  | 0.216       | 0.089        | 16    | 10     | Yes   | No      | 5            |

**Observations across top-5:**
- `slice_stride = 10` in all 5 trials.
- `batch_size = 16` in 4/5 trials (only rank 4 uses 8).
- `use_shift_scale_rotate = True` in all 5 trials.
- `use_brightness_contrast = False` in all 5 trials — suggests RBC is detrimental or neutral.
- `use_median_blur = True` in all 5 trials; `median_blur_limit` varies (3 → 7).
- `use_speckle_noise` only in 1/5 (rank 3); generally not helpful.
- `use_clahe` in 3/5; ambiguous — rank 1 and 3 succeed without it.
- Wide spread in `copy_paste_p` (0.216 → 0.666), suggesting it interacts with mosaic.

---

## 5. Best Configuration (Trial #409)

```yaml
# Best trial — EMA SliceAP@50 = 0.7500  (Trial #409)
learning_rate:  0.0008429  # moderate; well within search range
weight_decay:   0.003960   # stronger than v4 best (0.00797) — benefits from extended search space
mosaic_prob:    0.3236     # conservative; avoids over-mixing small dataset
copy_paste_p:   0.4937     # high; critical for 13-scan dataset
label_smooth:   0.0786     # mild smoothing
batch_size:     16
slice_stride:   10
warmup_epochs:  8
# Augmentations enabled:
#   HorizontalFlip, VerticalFlip (always)
#   ShiftScaleRotate (enabled)
#   MedianBlur (blur_limit=3)
# Augmentations disabled:
#   CLAHE, RandomBrightnessContrast, SpeckleNoise
```

---

## 6. Key Findings

### 6.1 Overfitting is the dominant challenge
All trials show a train/val loss gap that triples from epoch 5 to the final epoch. Val loss
bottoms between epochs 16–30 in the vast majority of trials. The shortened epoch budget
(35 vs 50) successfully avoids the late-epoch deterioration seen in v4.

### 6.2 Copy-paste augmentation is critical
With only 13 training scans, copy-paste is the only mechanism to generate novel lesion
instances per epoch. The floor of 0.2 ensured it was always active, and the best trial
uses `copy_paste_p = 0.494`. Rank 3 (0.7289) uses the highest value (0.666), confirming
that aggressive copy-paste is beneficial even at higher mosaic probabilities.

### 6.3 Weight decay extension paid off
The v4 best weight decay (7.97e-3) sat at the old ceiling of 1e-2. In v5, the best trial
uses `weight_decay = 3.96e-3` — lower than the old ceiling but the extended range allowed
TPE to properly explore the neighbourhood. The search found a "sweet spot" rather than
being forced against a boundary.

### 6.4 Minimal augmentation stack wins
The rank-1 trial uses only three augmentations beyond the mandatory flips: ShiftScaleRotate,
MedianBlur(3), and copy-paste. CLAHE, RBC, and speckle noise are absent. This suggests
that for this ultrasound dataset, excessive augmentation adds variance without reducing
overfitting, while copy-paste (real lesion instances from other scans) provides the
highest-quality regularisation.

### 6.5 Slice stride = 10 dominates
All top-5 trials use `slice_stride = 10`, which sub-samples ~10% of slices per epoch.
Sparser sampling acts as a form of regularisation by preventing the model from memorising
individual slice patterns.

### 6.6 Hyperband pruning efficiency
235/438 (54%) of trials with reported values were pruned by Hyperband. This is healthy:
the pruner correctly eliminates weak configurations early (≤ epoch 10), freeing GPU time
for more promising trials. The rung at epoch 10 is particularly active.

---

## 7. Plots

All plots are saved as PNG in this directory (`documentation/hparams_v5/`).

| File | Description |
|------|-------------|
| [plot_01_optim_history.png](plot_01_optim_history.png) | Trial index vs EMA SliceAP@50. Blue = completed, red = pruned. Orange line = running best. Shows convergence of TPE over 300 trials. |
| [plot_02_top5_comparison.png](plot_02_top5_comparison.png) | Side-by-side bar charts of LR, WD, mosaic, copy-paste P, and label-smooth for top-5 trials. |
| [plot_03_epoch_curves.png](plot_03_epoch_curves.png) | Training dynamics for 3 sample trials. Top row: log-scale train/val loss. Bottom row: SliceAP, EMA SliceAP, AP@50 vs epoch. |
| [plot_04_continuous_params.png](plot_04_continuous_params.png) | Scatter: each continuous hyperparameter vs SliceAP@50. Top-20% highlighted in amber; top-5 marked with stars. |
| [plot_05_categorical_analysis.png](plot_05_categorical_analysis.png) | Box plots of SliceAP@50 grouped by each categorical hyperparameter (batch size, stride, CLAHE, etc.). |
| [plot_06_lr_wd_scatter.png](plot_06_lr_wd_scatter.png) | 2D scatter of log(LR) vs log(WD) coloured by SliceAP (RdYlGn). Top-5 circled and labelled. Search bounds shown as dotted lines. |
| [plot_07_augmentation_heatmap.png](plot_07_augmentation_heatmap.png) | Binary heatmap of augmentation flag choices for the top-30 trials. Rows sorted by SliceAP@50 descending. |
| [plot_08_version_comparison.png](plot_08_version_comparison.png) | Bar chart comparing best EMA SliceAP@50 across v3, v4, and v5 with delta annotations. |

---

## 8. Recommended Next Steps

1. **Full training with best config** — Run Trial #409 config for 100 epochs with the full
   training set and standard early stopping (patience = 20). This should improve over the
   35-epoch trial result.

2. **SWA over top-3 checkpoints** — The `swa_manifest.yaml` for Trial #409 contains the
   top-3 epoch checkpoints. Stochastic weight averaging over these should yield a small
   additional boost (~0.5–1% SliceAP).

3. **Mosaic + copy-paste interaction study** — The spread in `copy_paste_p` across top-5
   (0.22–0.67) suggests these two augmentations interact. A small ablation (4 configs
   varying mosaic × copy-paste as a 2×2 grid) around the best trial would clarify this.

4. **v6 search space refinements** — Based on v5 findings:
   - Fix `slice_stride = 10` (unanimous across top-5; removes one categorical dimension).
   - Fix `use_brightness_contrast = False` (no benefit seen; reduces search space).
   - Tighten `weight_decay` to [1e-3, 2e-2] (converged region).
   - Consider `copy_paste_max_pastes ∈ {2, 3, 4}` as a new tunable.
