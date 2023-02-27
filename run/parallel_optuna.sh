#!/bin/bash

MAIN="/home/uz1/projects/GCN/GraphGym/full_pipe.py"
DATASET=$1
MAX_JOBS=${2:-2}
TAG=${3:-""}
LIMIT=${4:-""}
CUR_JOBS=0

while true; do
  if [ "$CUR_JOBS" -lt "$MAX_JOBS" ]; then
    echo "Job launched: $CUR_JOBS"
    if [ -z "$TAG" ]; then
        python $MAIN --dataset $DATASET &
    elif [ -z "$LIMIT" ]; then
        python $MAIN --dataset $DATASET --tag $TAG &
    else
        python $MAIN --dataset $DATASET --tag $TAG --limit $LIMIT &
    fi
    ((++CUR_JOBS))
    sleep 60
  else
    wait -n
    ((--CUR_JOBS))
  fi
done
