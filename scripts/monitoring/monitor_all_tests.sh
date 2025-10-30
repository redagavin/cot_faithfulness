#!/bin/bash

# Monitor all current test jobs
echo "=== Monitoring All Test Jobs ==="
echo "Started at: $(date)"
echo ""

# Job IDs to monitor
RERUN_JOBS=(2393718 2393719 2393720 2393721)
BHCS_RETEST_JOBS=(2394299 2394300)
ALL_JOBS=(${RERUN_JOBS[@]} ${BHCS_RETEST_JOBS[@]})

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
        echo "=== ALL TESTS COMPLETED ==="
        echo "Finished at: $(date)"
        echo ""
        echo "=== Final Status ==="
        for job in "${ALL_JOBS[@]}"; do
            echo "Job $job:"
            sacct -j $job --format=JobID,JobName,State,ExitCode,Elapsed | grep "^$job "
        done
        break
    fi

    timestamp=$(date +"%H:%M:%S")
    echo "[$timestamp] Still running: $running_count/6 jobs"

    sleep 180  # Check every 3 minutes
done
