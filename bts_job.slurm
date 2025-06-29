#!/bin/bash
#SBATCH --job-name=tf_job                # Job name
#SBATCH --output=bts.log           # Output log file
#SBATCH --error=bts.log             # Error log file
#SBATCH --gres=gpu:v100-32:1             # Request 1 V100-32GB GPU
#SBATCH --partition=GPU-shared           # Use GPU-shared partition
#SBATCH --time=4:00:00                   # 4-hour time limit
#SBATCH --mem=32G                        # Memory allocation (adjust if needed)
#SBATCH --cpus-per-task=4                # Number of CPU cores (adjust if needed)

# Load your TensorFlow module
module purge
module load AI/tensorflow_23.02-2.10.0-py3

# Run your script
python run_bts.py
