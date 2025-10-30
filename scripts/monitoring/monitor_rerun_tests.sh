#!/bin/bash

echo "=== Monitoring 4 Rerun Tests ==="
echo "Started at: $(date)"
echo ""

JOBS=(2393718 2393719 2393720 2393721)
NAMES=("test_diagnosis_arena" "test_medxpertqa" "test_diagnosis_arena_baseline" "test_medxpertqa_baseline")

while true; do
    sleep 180  # Check every 3 minutes

    RUNNING=0
    for i in "${!JOBS[@]}"; do
        JOB=${JOBS[$i]}
        if squeue -j $JOB 2>/dev/null | grep -q $JOB; then
            RUNNING=$((RUNNING + 1))
        fi
    done

    if [ $RUNNING -eq 0 ]; then
        echo "All test jobs completed at $(date)"
        break
    fi

    echo "[$( date '+%H:%M:%S')] Still running: $RUNNING jobs"
done
