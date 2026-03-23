#!/bin/bash
#SBATCH --job-name=yoloe_eval_v5_ciou_tal_ohem_05
#SBATCH --account=tenomix
#SBATCH --partition=rtx6000_b2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/fs01/home/dhaneshr/code/obj-det-test/slurm-script/logs/yoloe_eval_v5_ciou_tal_ohem_05_%j.out
#SBATCH --error=/fs01/home/dhaneshr/code/obj-det-test/slurm-script/logs/yoloe_eval_v5_ciou_tal_ohem_05_%j.err

PROJECT_DIR="/fs01/home/dhaneshr/code/obj-det-test"
WEIGHTS="${PROJECT_DIR}/checkpoints/best_model_v5_ciou_tal_ohem_20260320_134819.pth"
CONFIG_FILE="train/config/best_hyperparams_v5.yaml"
OUT_DIR="${PROJECT_DIR}/inference/test_results_v5_ciou_tal_ohem_conf05_iou05"

echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"

cd "$PROJECT_DIR" || exit 1
source .venv/bin/activate

echo "Evaluating checkpoint: $WEIGHTS"
python inference/test_eval.py \
    --weights    "$WEIGHTS" \
    --config     "$CONFIG_FILE" \
    --split      test \
    --conf-thresh 0.5 \
    --iou-thresh  0.5 \
    --out-dir    "$OUT_DIR"

echo "Job completed at: $(date)"
