#!/bin/bash
#SBATCH --job-name=test_diagnosis_arena
#SBATCH --output=logs/test_diagnosis_arena_%j.out
#SBATCH --error=logs/test_diagnosis_arena_%j.err
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
echo "TEST RUN - DiagnosisArena Analysis (10 samples)"
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
echo "Starting DiagnosisArena TEST analysis (10 samples)..."
echo "============================================================"
cd /scratch/yang.zih/cot

python src/test_diagnosis_arena.py

# Check if output file was created
echo ""
echo "============================================================"
echo "Checking Output"
echo "============================================================"
if [ -f "tests/test_results/test_diagnosis_arena_results.xlsx" ]; then
    echo "✓ Test completed successfully!"
    echo "✓ Output file: tests/test_results/test_diagnosis_arena_results.xlsx"
    echo "✓ File size: $(ls -lh tests/test_results/test_diagnosis_arena_results.xlsx | awk '{print $5}')"
    echo ""
    echo "Please review the output file to verify:"
    echo "  1. Gender detection worked correctly"
    echo "  2. Bidirectional mapping (female/male labels)"
    echo "  3. Answer extraction (A/B/C/D)"
    echo "  4. Judge assessments are 'UNFAITHFUL' or 'EXPLICIT BIAS'"
    echo "  5. Correctness tracking (female_correct, male_correct)"
    echo "  6. Summary statistics (flip analysis, same answer analysis)"
else
    echo "✗ Warning: Output file not found!"
    echo "Check the error log for issues."
fi

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "============================================================"
