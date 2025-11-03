#!/bin/bash
#SBATCH --job-name=diagnosis_arena
#SBATCH --output=logs/diagnosis_arena_%j.out
#SBATCH --error=logs/diagnosis_arena_%j.err
#SBATCH --time=3-00:00:00
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
echo "FULL RUN - DiagnosisArena Analysis (All Filtered Cases)"
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

# Run the full analysis
echo "============================================================"
echo "Starting DiagnosisArena FULL analysis..."
echo "============================================================"
echo "Expected: ~600-700 cases after filtering"
echo "Estimated runtime: 10-15 hours"
echo ""
cd /scratch/yang.zih/cot

python src/diagnosis_arena_analysis.py

# Check if output file was created
echo ""
echo "============================================================"
echo "Checking Output"
echo "============================================================"
if [ -f "results/diagnosis_arena_results.xlsx" ]; then
    echo "✓ Analysis completed successfully!"
    echo "✓ Output file: results/diagnosis_arena_results.xlsx"
    echo "✓ File size: $(ls -lh results/diagnosis_arena_results.xlsx | awk '{print $5}')"
    echo ""
    echo "Results include:"
    echo "  - Analysis Results sheet (all responses and answers)"
    echo "  - Summary statistics (match rates, flip rates, correctness analysis)"
    echo "  - Gender mapping reference"
else
    echo "✗ Warning: Output file not found!"
    echo "Check the error log for issues."
fi

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "============================================================"
