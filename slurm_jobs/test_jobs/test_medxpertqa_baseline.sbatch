#!/bin/bash
#SBATCH --job-name=test_medxpertqa_baseline
#SBATCH --output=logs/test_medxpertqa_baseline_%j.out
#SBATCH --error=logs/test_medxpertqa_baseline_%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=frink
#SBATCH --gres=gpu:quadro:1
#SBATCH --mem=64G

echo "========================================="
echo "MedXpertQA Baseline Analysis - Test Run (10 samples)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
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

# Create test script
cat > test_medxpertqa_baseline_script.py << 'EOF'
from medxpertqa_baseline_analysis import MedXpertQABaselineAnalyzer

analyzer = MedXpertQABaselineAnalyzer()
analyzer.run_complete_analysis(sample_size=10)  # Only 10 samples for testing
EOF

echo "Running MedXpertQA baseline analysis (test)..."
python src/test_medxpertqa_baseline_script.py

echo ""
echo "========================================="
echo "Test run completed"
echo "End time: $(date)"
echo "Results saved to: tests/test_results/test_medxpertqa_baseline_results.xlsx"
echo "========================================="
