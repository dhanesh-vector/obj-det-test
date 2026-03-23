#!/bin/bash
#SBATCH --job-name=yoloe_train_v5_ciou_tal_ohem
#SBATCH --account=tenomix
#SBATCH --partition=rtx6000_b2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/fs01/home/dhaneshr/code/obj-det-test/slurm-script/logs/yoloe_train_v5_ciou_tal_ohem_%j.out
#SBATCH --error=/fs01/home/dhaneshr/code/obj-det-test/slurm-script/logs/yoloe_train_v5_ciou_tal_ohem_%j.err

PROJECT_DIR="/fs01/home/dhaneshr/code/obj-det-test"
LOGS_DIR="${PROJECT_DIR}/slurm-script/logs"

mkdir -p "$LOGS_DIR"

echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"

cd "$PROJECT_DIR" || exit 1

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Starting training — v5 hyperparams + CIoU box loss + Task-Aligned Assignment + Hard Negative Mining..."
python train/train.py \
    --config train/config/best_hyperparams_v5.yaml \
    --run_tag v5_ciou_tal_ohem

echo "Job completed at: $(date)"
