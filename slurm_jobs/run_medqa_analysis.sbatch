#!/bin/bash
#SBATCH --job-name=medqa_gender
#SBATCH --output=logs/medqa_gender_%j.out
#SBATCH --error=logs/medqa_gender_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=frink
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# Activate conda environment
source ~/.bashrc
conda activate cot

# Print job info
echo "========================================"
echo "MedQA Gender Analysis - FULL RUN"
echo "Dataset: GBaker/MedQA-USMLE-4-options-hf"
echo "Models: Olmo2-7B, DeepSeek-R1-0528"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# Check GPU
nvidia-smi

echo ""
echo "Starting full analysis..."
echo "Expected runtime: 4-6 hours"
echo ""

# Run full analysis
python src/medqa_analysis.py

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ANALYSIS COMPLETED SUCCESSFULLY"
    echo "Results saved to: results/medqa_analysis_results.xlsx"
else
    echo "❌ ANALYSIS FAILED OR INCOMPLETE"
fi

echo "========================================"

exit $EXIT_CODE
