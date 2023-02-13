#!/usr/bin/env bash

CONFIG=example_graph_cluster
GRID=example_cluster
REPEAT=1
MAX_JOBS=3
MAIN=main

# generate configs (after controlling computational budget)
cd /home/uz1/projects/GCN/GraphGym/run
# please remove --config_budget, if don't control computational budget
python /home/uz1/projects/GCN/GraphGym/run/configs_gen.py --config configs/pyg/${CONFIG}.yaml \
  --grid grids/pyg/${GRID}.txt \
  --out_dir configs
#python configs_gen.py --config configs/ChemKG/${CONFIG}.yaml --config_budget configs/ChemKG/${CONFIG}.yaml --grid grids/ChemKG/${GRID}.txt --out_dir configs
# run batch of configs
# Args: config_dir, num of repeats, max jobs running
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $MAIN
# rerun missed / stopped experiments
 bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $MAIN
# rerun missed / stopped experiments
#bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $MAIN

# aggregate results for the batch
python /home/uz1/projects/GCN/GraphGym/run/agg_batch.py --dir results/${CONFIG}_grid_${GRID}
