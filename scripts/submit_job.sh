#!/bin/bash
# Convenience script to submit the BHCS analysis job

echo "Submitting BHCS Analysis job to SLURM..."
echo "Working directory: $(pwd)"

# Make the sbatch script executable
chmod +x slurm_jobs/run_bhcs_analysis.sbatch

# Submit the job
JOB_ID=$(sbatch --parsable slurm_jobs/run_bhcs_analysis.sbatch)

if [ $? -eq 0 ]; then
    echo "Job submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo ""
    echo "To monitor the job:"
    echo "  squeue -u $USER"
    echo "  squeue -j $JOB_ID"
    echo ""
    echo "To view logs in real-time:"
    echo "  tail -f logs/bhcs_analysis_${JOB_ID}.out"
    echo "  tail -f logs/bhcs_analysis_${JOB_ID}.err"
    echo ""
    echo "To cancel the job if needed:"
    echo "  scancel $JOB_ID"
else
    echo "Error: Failed to submit job!"
    exit 1
fi