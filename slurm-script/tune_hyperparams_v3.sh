#!/bin/bash
#SBATCH --job-name=yoloe_tune_v3
#SBATCH --account=tenomix
#SBATCH --partition=rtx6000_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-19                 # 20 parallel workers  →  20 × 15 = 300 trials total
#SBATCH --output=/fs01/home/dhaneshr/code/obj-det-test/slurm-script/logs/yoloe_tune_v3_%A_%a.out
#SBATCH --error=/fs01/home/dhaneshr/code/obj-det-test/slurm-script/logs/yoloe_tune_v3_%A_%a.err

# ─── Config ───────────────────────────────────────────────────────────────────
PROJECT_DIR="/fs01/home/dhaneshr/code/obj-det-test"
CONFIG_FILE="${1:-train/config/default.yaml}"   # pass as first arg or use default
STUDY_NAME="yoloe_hparam_search_v3"
N_TRIALS=15          # trials per worker  (total = N_TRIALS × n_array_tasks = 300)
EPOCHS=50            # max epochs per trial (early stopping cuts most short)
EARLY_STOP=25        # patience per trial
MIN_DELTA=0.005      # min AP@50 improvement to reset early-stopping counter
SAVE_TOP_K=5         # how many ranked configs to write as YAML

# Shared SQLite DB on NFS — all workers read/write the same study
STORAGE_PATH="${PROJECT_DIR}/optuna_studies/${STUDY_NAME}.db"
STORAGE_URL="sqlite:///${STORAGE_PATH}"
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p "${PROJECT_DIR}/slurm-script/logs"
mkdir -p "${PROJECT_DIR}/optuna_studies"

echo "Job array  : ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "Node       : ${SLURM_NODELIST}"
echo "Started at : $(date)"
echo "Config     : ${CONFIG_FILE}"
echo "Study      : ${STUDY_NAME} @ ${STORAGE_URL}"
echo "Trials     : ${N_TRIALS} (this worker)"

cd "$PROJECT_DIR" || exit 1
source .venv/bin/activate

# Stagger workers so they don't all hit the SQLite DB simultaneously on startup
JITTER="${SLURM_ARRAY_TASK_ID}"

python train/tune_hyperparams_v3.py \
    --config                    "$CONFIG_FILE" \
    --study-name                "$STUDY_NAME" \
    --storage                   "$STORAGE_URL" \
    --n-trials                  "$N_TRIALS" \
    --epochs                    "$EPOCHS" \
    --early-stopping-patience   "$EARLY_STOP" \
    --early-stopping-min-delta  "$MIN_DELTA" \
    --save-top-k                "$SAVE_TOP_K" \
    --output-config-dir         "train/config" \
    --worker-jitter             "$JITTER"

echo "Task ${SLURM_ARRAY_TASK_ID} completed at: $(date)"
