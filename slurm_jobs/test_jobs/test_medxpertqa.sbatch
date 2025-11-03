#!/bin/bash
#SBATCH --job-name=test_medxpertqa
#SBATCH --output=logs/test_medxpertqa_%j.out
#SBATCH --error=logs/test_medxpertqa_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=frink
#SBATCH --gres=gpu:quadro:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yang.zih@northeastern.edu

# Print job information
echo "============================================================"
echo "TEST RUN - MedXpertQA Analysis (10 sample cases)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Source .bashrc
source ~/.bashrc

# Activate conda environment
echo "Activating conda environment 'cot'..."
conda activate cot

# Check the environment
echo ""
echo "Python path:"
which python
echo ""
echo "Current conda environment:"
echo $CONDA_DEFAULT_ENV
echo ""

# Install/update requirements if needed
echo "Installing/updating dependencies..."
uv pip install -r requirements.txt

# Print GPU information
echo ""
echo "============================================================"
echo "GPU Information:"
echo "============================================================"
nvidia-smi
echo ""

# Print Python and PyTorch information
echo "============================================================"
echo "Python and PyTorch Information:"
echo "============================================================"
echo "Python version: $(python --version)"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null || echo "No GPUs detected"
echo ""

# Run the test
echo "============================================================"
echo "Starting MedXpertQA TEST analysis..."
echo "============================================================"
echo "Expected: ~10 cases for initial validation"
echo "Estimated runtime: 10-20 minutes"
echo ""
cd /scratch/yang.zih/cot

# Create temporary test script with sample_size=10
cat > test_medxpertqa_script.py << 'EOF'
from medxpertqa_analysis import MedXpertQAAnalyzer

analyzer = MedXpertQAAnalyzer()
analyzer.run_complete_analysis(sample_size=10)

print("\nTest Results Summary:")
print(f"- Total cases processed: {len(analyzer.results)}")
models_used = [name for name in analyzer.get_models_config().keys() if any(f'{name}_answers_match' in result for result in analyzer.results)]
print(f"- Models used: {models_used}")
print(f"- Results saved to tests/test_results/test_medxpertqa_results.xlsx")
EOF

python src/test_medxpertqa_script.py

# Clean up temporary script
rm test_medxpertqa_script.py

# Check if output file was created
echo ""
echo "============================================================"
echo "Checking Output"
echo "============================================================"
if [ -f "tests/test_results/test_medxpertqa_results.xlsx" ]; then
    echo "✓ Test completed successfully!"
    echo "✓ Output file: tests/test_results/test_medxpertqa_results.xlsx"
    echo "✓ File size: $(ls -lh tests/test_results/test_medxpertqa_results.xlsx | awk '{print $5}')"
    echo ""
    echo "Results include:"
    echo "  - Analysis Results sheet (all responses and answers)"
    echo "  - Summary statistics (match rates, flip rates, correctness analysis)"
    echo "  - Gender mapping reference"
    echo ""
    echo "If test looks good, proceed with full run using run_medxpertqa.sbatch"
else
    echo "✗ Warning: Output file not found!"
    echo "Check the error log for issues."
fi

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "============================================================"
