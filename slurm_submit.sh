#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1  # Adjust GPU type and count as needed
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00  # Adjust the time limit
#SBATCH --job-name=colpali_distill_test
#SBATCH --output=colpali_distill_slurm_%j.out  # Output file
#SBATCH --error=colpali_distill_slurm_%j.err   # Error file

# Navigate to your project directory
cd /project2/ywang234_1595/colpali-distill/colpali-distill

# Activate the virtual environment
source venv/bin/activate

pip install colpali-engine[train] dotenv hf-transfer wandb

# Insert WandB and HF_Token
# WANDB_API_KEY = 
# HF_TOKEN = 

# Run your training script
python -u train_distill_script.py train_config.yaml