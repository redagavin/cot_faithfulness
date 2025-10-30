#!/bin/bash
#SBATCH --job-name=test_medqa_baseline
#SBATCH --output=logs/test_medqa_baseline_%j.out
#SBATCH --error=logs/test_medqa_baseline_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=177huntington
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# Activate conda environment
source ~/.bashrc
conda activate cot

# Print job info
echo "========================================"
echo "MedQA Baseline Analysis - TEST (10 samples)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# Check GPU
nvidia-smi

echo ""
echo "Starting test run..."
echo ""

# Run test
python src/test_medqa_baseline_analysis.py

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TEST COMPLETED"
    echo "Please validate results in results/test_medqa_baseline_results.xlsx"
else
    echo "❌ TEST FAILED"
fi

echo "========================================"

exit $EXIT_CODE
