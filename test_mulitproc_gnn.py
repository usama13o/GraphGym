from graphgym.config import cfg, dump_cfg, load_cfg, set_run_dir, set_out_dir

from graphgym.cmd_args import parse_args
from graphgym.config import cfg, dump_cfg, load_cfg, set_run_dir, set_out_dir
from graphgym.loader import create_dataset, create_loader
from graphgym.logger_optuna import create_logger, setup_printing
from graphgym.model_builder import create_model
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.register import train_dict
from graphgym.train_optuna import train
from graphgym.utils.agg_runs import agg_runs
from graphgym.utils.comp_budget import params_count
from graphgym.utils.device import auto_select_device
from torch_geometric import seed_everything
import optuna
from optuna.trial import TrialState
import argparse
import datetime
import logging
import sys
import os
import subprocess
import torch
import concurrent.futures
parser = argparse.ArgumentParser(description=f"fullpipline test multi proc ")
parser.add_argument('--dataset', type=str, default=None,required=False)
parser.add_argument('--tag', type=str, default="",required=False)
parser.add_argument('--limit', type=int, default=11000,required=False)
args = parser.parse_args() 
args.opts = []
dataset = 'pathmnist'
patch_size = 24
num_nodes = 8
image_size = 192
# run graphgym - on three different cfgs with graph generated above 
g_data_path = f"/home/uz1/projects/GCN/graph_data/graph-data---{dataset}-{patch_size}-{num_nodes}-{image_size}-UC_False.h5" 
MAIN = "/home/uz1/projects/GCN/GraphGym/run/main.py"
args.cfg_file = "/home/uz1/projects/GCN/GraphGym/run/configs/pyg/example_graph_cluster_copy.yaml"

load_cfg(cfg, args)
cfg.dataset.dir = g_data_path
# run graphgym on cf
cfg.gnn.layer_type = "modgeneraledgeconv"

set_run_dir(cfg.out_dir)
setup_printing()
# Set configurations for each run
cfg.seed = cfg.seed + 1
seed_everything(cfg.seed)
auto_select_device()
# Set machine learning pipeline
datasets = create_dataset()
loaders = create_loader(datasets)
loggers = create_logger()
model = create_model()
optimizer = create_optimizer(model.parameters())
scheduler = create_scheduler(optimizer)
# Print model info
logging.info(model)
logging.info(cfg)
cfg.params = params_count(model)
logging.info('Num parameters: %s', cfg.params)
# Start training

# run graphgym on cfg2
cfg.gnn.layer_type = "sageconv"
model2 = create_model()
loggers2 = create_logger()
optimizer2 = create_optimizer(model.parameters())
scheduler2 = create_scheduler(optimizer)
# Print model info
logging.info(model)
logging.info(cfg)
cfg.params = params_count(model)
logging.info('Num parameters: %s', cfg.params)
# Start training

# run graphgym on 
cfg.gnn.layer_type = "gatconv"
model3 = create_model()
loggers3 = create_logger()
optimizer3 = create_optimizer(model.parameters())
scheduler3 = create_scheduler(optimizer)
# Print model info
logging.info(model)
logging.info(cfg)
cfg.params = params_count(model)
logging.info('Num parameters: %s', cfg.params)
# Start training

# get accuracy average - make sure they're all floats
trial = None
print("starting a thread pool .. . ")
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    # Submit the functions to the pool and get the futures
    future_1 = executor.submit(train,loggers, loaders, model, optimizer, scheduler, trial)
    future_2 = executor.submit(train,loggers2, loaders, model2, optimizer2, scheduler2, trial)
    future_3 = executor.submit(train,loggers3, loaders, model3, optimizer3, scheduler3, trial)
result_1 = future_1.result()
result_2 = future_2.result()
result_3 = future_3.result()
average_result = (result_1 + result_2 + result_3) / 3
print("average result is ", average_result)
"""
train(loggers, loaders, model, optimizer, scheduler, trial)
import concurrent.futures

def function_1():
    # do some work
    return result_1

def function_2():
    # do some work
    return result_2

def function_3():
    # do some work
    return result_3

# Create a thread pool with 3 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    # Submit the functions to the pool and get the futures
    future_1 = executor.submit(function_1)
    future_2 = executor.submit(function_2)
    future_3 = executor.submit(function_3)

# Get the results from the futures and average them
result_1 = future_1.result()
result_2 = future_2.result()
result_3 = future_3.result()
average_result = (result_1 + result_2 + result_3) / 3

"""