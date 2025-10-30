#!/bin/bash

echo "=== Test Job Monitoring (Round 2) ==="
echo "Started at: $(date)"
echo ""

JOBS=(2384774 2384775 2384776 2384777 2384778 2384779)
NAMES=("test_bhcs_analysis" "test_diagnosis_arena" "test_medxpertqa" "test_bhcs_baseline" "test_diagnosis_arena_baseline" "test_medxpertqa_baseline")

while true; do
    echo "=== Status Check: $(date) ==="
    echo ""

    all_done=true
    for i in "${!JOBS[@]}"; do
        JOB=${JOBS[$i]}
        NAME=${NAMES[$i]}

        # Check if job is still in queue
        STATUS=$(squeue -j $JOB 2>/dev/null | grep $JOB | awk '{print $5}')

        if [ -z "$STATUS" ]; then
            echo "[$JOB] $NAME: COMPLETED"

            # Check for errors in error file
            ERR_FILE=$(ls -t ${NAME}_*.err 2>/dev/null | head -1)
            if [ -f "$ERR_FILE" ]; then
                ERR_SIZE=$(wc -l < "$ERR_FILE")
                if [ $ERR_SIZE -gt 0 ]; then
                    echo "  WARNING: Error file has $ERR_SIZE lines"
                fi
            fi
        else
            echo "[$JOB] $NAME: $STATUS"
            all_done=false
        fi
    done

    echo ""

    if [ "$all_done" = true ]; then
        echo "=== All test jobs completed! ==="
        echo "Completed at: $(date)"
        break
    fi

    echo "Waiting 60 seconds before next check..."
    echo "----------------------------------------"
    sleep 60
done
