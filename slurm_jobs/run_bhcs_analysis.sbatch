#!/bin/bash
#SBATCH --job-name=bhcs_analysis
#SBATCH --output=logs/bhcs_analysis_%j.out
#SBATCH --error=logs/bhcs_analysis_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=frink
#SBATCH --gres=gpu:quadro:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yang.zih@northeastern.edu

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Source .bashrc
source ~/.bashrc

# Activate conda environment
echo "Activating conda environment 'cot'..."
conda activate cot

# Check the environment
# Check environment information
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
echo "GPU Information:"
nvidia-smi

# Print Python and PyTorch information
echo "Python version: $(python --version)"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null || echo "No GPUs detected"

# Run the analysis
echo "Starting BHCS analysis..."
cd /scratch/yang.zih/cot_faithfulness

# Run with full dataset (remove sample_size parameter from main() for complete analysis)
python src/bhcs_analysis.py

# Check if output file was created
if [ -f "results/bhcs_analysis_results.xlsx" ]; then
    echo "Analysis completed successfully!"
    echo "Output file size: $(ls -lh results/bhcs_analysis_results.xlsx | awk '{print $5}')"
else
    echo "Warning: Output file not found!"
fi

echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"