#!/bin/bash

# Monitor all production jobs
echo "=== PRODUCTION JOBS MONITOR ==="
echo "Started at: $(date)"
echo ""

# Production job IDs
BHCS_JOBS=(2396633 2396634)
DIAG_JOBS=(2396635 2396636)
MEDX_JOBS=(2396637 2396638)
ALL_JOBS=(${BHCS_JOBS[@]} ${DIAG_JOBS[@]} ${MEDX_JOBS[@]})

# Test job
TEST_JOB=2396095

while true; do
    # Count running jobs
    running_count=0
    for job in "${ALL_JOBS[@]}"; do
        if squeue -j $job &> /dev/null; then
            ((running_count++))
        fi
    done

    if [ $running_count -eq 0 ]; then
        echo ""
        echo "=== ALL PRODUCTION JOBS COMPLETED ==="
        echo "Finished at: $(date)"
        echo ""
        echo "=== Final Status ==="
        for job in "${ALL_JOBS[@]}"; do
            sacct -j $job --format=JobID,JobName,State,ExitCode,Elapsed | grep "^$job "
        done

        echo ""
        echo "=== Test Job Status ==="
        if squeue -j $TEST_JOB &> /dev/null; then
            echo "Test job $TEST_JOB: STILL RUNNING"
        else
            sacct -j $TEST_JOB --format=JobID,JobName,State,ExitCode,Elapsed | grep "^$TEST_JOB "
        fi
        break
    fi

    timestamp=$(date +"%H:%M:%S")
    echo "[$timestamp] Running: $running_count/6 production jobs"

    # Show individual status
    echo "  BHCS: $(squeue -j 2396633,2396634 -h 2>/dev/null | wc -l)/2"
    echo "  DiagnosisArena: $(squeue -j 2396635,2396636 -h 2>/dev/null | wc -l)/2"
    echo "  MedXpertQA: $(squeue -j 2396637,2396638 -h 2>/dev/null | wc -l)/2"
    echo "  Test (2396095): $(squeue -j 2396095 -h 2>/dev/null | wc -l)/1"

    sleep 600  # Check every 10 minutes
done
