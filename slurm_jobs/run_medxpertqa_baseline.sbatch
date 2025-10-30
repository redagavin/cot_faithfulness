#!/bin/bash
#SBATCH --job-name=medxpertqa_baseline
#SBATCH --output=logs/medxpertqa_baseline_%j.out
#SBATCH --error=logs/medxpertqa_baseline_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --partition=177huntington
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

echo "========================================="
echo "MedXpertQA Baseline Analysis - Full Run"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start time: $(date)"
echo "Time limit: 48 hours (2 days)"
echo ""

# Source .bashrc
source ~/.bashrc

# Activate conda environment
echo "Activating conda environment 'cot'..."
conda activate cot

# Install/update dependencies if needed
echo "Installing/updating dependencies..."
uv pip install -r requirements.txt

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

echo "Running MedXpertQA baseline analysis (full dataset)..."
echo "Estimated runtime: 24-36 hours (1,505 analyzable cases)"
echo ""

python src/medxpertqa_baseline_analysis.py

echo ""
echo "========================================="
echo "Full run completed"
echo "End time: $(date)"
echo "Results saved to: results/medxpertqa_baseline_results.xlsx"
echo "========================================="
