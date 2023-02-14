#!/bin/bash

MAIN="/home/uz1/projects/GCN/GraphGym/run/main_optune.py"

MAX_JOBS=4

CUR_JOBS=0
while true; do
  if [ "$CUR_JOBS" -lt "$MAX_JOBS" ]; then
    echo "Job launched: $CUR_JOBS"
    python $MAIN &
    ((++CUR_JOBS))
  else
    wait -n
    ((--CUR_JOBS))
  fi
done
