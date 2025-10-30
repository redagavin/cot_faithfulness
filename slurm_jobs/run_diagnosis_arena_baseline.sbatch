#!/bin/bash
#SBATCH --job-name=diagnosis_baseline
#SBATCH --output=logs/diagnosis_arena_baseline_%j.out
#SBATCH --error=logs/diagnosis_arena_baseline_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=177huntington
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

echo "========================================="
echo "DiagnosisArena Baseline Analysis - Full Run"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start time: $(date)"
echo "Time limit: 24 hours"
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

echo "Running DiagnosisArena baseline analysis (full dataset)..."
echo "Estimated runtime: 12-18 hours (605 analyzable cases)"
echo ""

python src/diagnosis_arena_baseline_analysis.py

echo ""
echo "========================================="
echo "Full run completed"
echo "End time: $(date)"
echo "Results saved to: results/diagnosis_arena_baseline_results.xlsx"
echo "========================================="
