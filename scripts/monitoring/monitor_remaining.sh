#!/bin/bash
for i in {1..20}; do
  count=$(squeue -j 2382709,2382710 --noheader 2>/dev/null | wc -l)
  if [ "$count" -eq 0 ]; then
    echo "All remaining jobs completed!"
    break
  fi
  echo "Check $i: $count jobs running at $(date +%H:%M:%S)"
  sleep 20
done
echo "Done at $(date +%H:%M:%S)"
