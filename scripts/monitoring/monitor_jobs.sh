#!/bin/bash
# Monitor test jobs until completion

for i in {1..40}; do
  count=$(squeue -j 2382708,2382709,2382710 --noheader 2>/dev/null | wc -l)
  if [ "$count" -eq 0 ]; then
    echo "All jobs completed!"
    break
  fi
  echo "Check $i: $count jobs still running at $(date +%H:%M:%S)"
  squeue -j 2382708,2382709,2382710 --noheader 2>/dev/null | awk '{print "  Job", $1, $3, $5}'
  sleep 20
done

echo ""
echo "Final check at $(date)"
squeue -j 2382708,2382709,2382710
