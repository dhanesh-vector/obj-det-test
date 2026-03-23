#!/bin/bash
#SBATCH --job-name=yoloe_tune_v5
#SBATCH --account=tenomix
#SBATCH --partition=rtx6000_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-19                 # 20 parallel workers  →  20 × 15 = 300 trials total
#SBATCH --output=/fs01/home/dhaneshr/code/obj-det-test/slurm-script/logs/yoloe_tune_v5_%A_%a.out
#SBATCH --error=/fs01/home/dhaneshr/code/obj-det-test/slurm-script/logs/yoloe_tune_v5_%A_%a.err

# ─── Config ───────────────────────────────────────────────────────────────────
PROJECT_DIR="/fs01/home/dhaneshr/code/obj-det-test"
CONFIG_FILE="${1:-train/config/default.yaml}"
STUDY_NAME="yoloe_hparam_search_v5"
N_TRIALS=15          # trials per worker  (total = N_TRIALS × array_size = 300)
EPOCHS=35            # v5: reduced from 50 — val loss bottoms at ep 16–30 in v4;
                     # ep 35–50 only deepens the train/val gap with no SliceAP gain
EARLY_STOP=8         # v5: tightened from 15 — EMA smoothing (α=0.3) already absorbs
                     # single-epoch dips; 8 epochs is sufficient tolerance
MIN_DELTA=0.005      # min EMA SliceAP@50 improvement to reset counter
EMA_ALPHA=0.3        # SliceAP@50 EMA smoothing factor
SAVE_TOP_K=5         # ranked YAML configs to write at the end
CKPT_TOP_K=3         # best checkpoints to keep per trial (for SWA averaging)
CKPT_MIN_EMA=0.6     # only save checkpoints for trials that reach this EMA SliceAP@50
CKPT_DIR="${PROJECT_DIR}/checkpoints/hparam_v5"

# Shared SQLite DB on NFS — all workers read/write the same Optuna study
STORAGE_PATH="${PROJECT_DIR}/optuna_studies/${STUDY_NAME}.db"
STORAGE_URL="sqlite:///${STORAGE_PATH}"
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p "${PROJECT_DIR}/slurm-script/logs"
mkdir -p "${PROJECT_DIR}/optuna_studies"
mkdir -p "${CKPT_DIR}"

echo "Job array    : ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "Node         : ${SLURM_NODELIST}"
echo "Started at   : $(date)"
echo "Config       : ${CONFIG_FILE}"
echo "Study        : ${STUDY_NAME} @ ${STORAGE_URL}"
echo "Trials       : ${N_TRIALS} (this worker)"
echo "Epochs/trial : ${EPOCHS} (max, EMA early stopping patience=${EARLY_STOP})"
echo "Checkpoints  : top-${CKPT_TOP_K} per trial (min EMA=${CKPT_MIN_EMA}) → ${CKPT_DIR}"

cd "$PROJECT_DIR" || exit 1
source .venv/bin/activate

# Stagger workers by task ID seconds to avoid simultaneous DB creation
JITTER="${SLURM_ARRAY_TASK_ID}"

python train/tune_hyperparams_v5.py \
    --config                    "$CONFIG_FILE" \
    --study-name                "$STUDY_NAME" \
    --storage                   "$STORAGE_URL" \
    --n-trials                  "$N_TRIALS" \
    --epochs                    "$EPOCHS" \
    --early-stopping-patience   "$EARLY_STOP" \
    --early-stopping-min-delta  "$MIN_DELTA" \
    --ema-alpha                 "$EMA_ALPHA" \
    --save-top-k                "$SAVE_TOP_K" \
    --ckpt-top-k                "$CKPT_TOP_K" \
    --ckpt-min-ema              "$CKPT_MIN_EMA" \
    --output-config-dir         "train/config" \
    --checkpoint-dir            "$CKPT_DIR" \
    --worker-jitter             "$JITTER"

echo "Task ${SLURM_ARRAY_TASK_ID} completed at: $(date)"
