#!/bin/bash
#SBATCH --job-name=validate_qwq_medxpertqa
#SBATCH --output=logs/validate_qwq_medxpertqa_%j.out
#SBATCH --error=logs/validate_qwq_medxpertqa_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=177huntington
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# Activate conda environment
source ~/.bashrc
conda activate cot

# Print job info
echo "========================================"
echo "QwQ-32B MedXpertQA Validation"
echo "Model: Qwen/QwQ-32B-Preview"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo ""
echo "Expected accuracy: 18.00%"
echo "Tolerance: ±2%"
echo "========================================"
echo ""

# Check GPU
nvidia-smi

echo ""
echo "Starting validation..."
echo ""

# Run validation
python src/validate_qwq_medxpertqa.py

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ VALIDATION PASSED"
else
    echo "❌ VALIDATION FAILED OR INCOMPLETE"
fi

echo "========================================"

exit $EXIT_CODE
