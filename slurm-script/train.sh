#!/bin/bash
#SBATCH --job-name=yoloe_train
#SBATCH --account=tenomix
#SBATCH --partition=rtx6000_b2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/fs01/home/dhaneshr/code/obj-det-test/slurm-script/logs/yoloe_train_%j.out
#SBATCH --error=/fs01/home/dhaneshr/code/obj-det-test/slurm-script/logs/yoloe_train_%j.err

# Define absolute paths
PROJECT_DIR="/fs01/home/dhaneshr/code/obj-det-test"
LOGS_DIR="${PROJECT_DIR}/slurm-script/logs"


# Ensure absolute output directory exists
mkdir -p "$LOGS_DIR"

echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"

# Move to the project root
cd "$PROJECT_DIR" || exit 1

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Execute the training script
echo "Starting training..."

CONFIG_FILE=${1:-train/config/default.yaml} # specify which config file to use here
echo "Using config file: $CONFIG_FILE"
python train/train.py --config "$CONFIG_FILE"

echo "Job completed at: $(date)"
