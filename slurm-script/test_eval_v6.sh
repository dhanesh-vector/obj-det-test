#!/bin/bash
#SBATCH --job-name=yoloe_eval_v6
#SBATCH --account=tenomix
#SBATCH --partition=rtx6000_b2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/fs01/home/dhaneshr/code/obj-det-test/slurm-script/logs/yoloe_eval_v6_%j.out
#SBATCH --error=/fs01/home/dhaneshr/code/obj-det-test/slurm-script/logs/yoloe_eval_v6_%j.err

PROJECT_DIR="/fs01/home/dhaneshr/code/obj-det-test"
CONFIG_FILE="train/config/best_hyperparams_v6.yaml"
OUT_DIR="${PROJECT_DIR}/inference/test_results_v6"

echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"

cd "$PROJECT_DIR" || exit 1
source .venv/bin/activate

# Find the most recently saved best_model checkpoint
WEIGHTS=$(ls -t "${PROJECT_DIR}/checkpoints"/best_model_*.pth 2>/dev/null | head -1)
if [[ -z "$WEIGHTS" ]]; then
    echo "ERROR: No best_model_*.pth found in checkpoints/" >&2
    exit 1
fi
echo "Using checkpoint: $WEIGHTS"

python inference/test_eval.py \
    --weights   "$WEIGHTS" \
    --config    "$CONFIG_FILE" \
    --split     test \
    --out-dir   "$OUT_DIR"

echo "Job completed at: $(date)"
