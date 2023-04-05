import logging
import os

import torch
from torch_geometric import seed_everything
import datetime
# from graphgym.cmd_args import parse_args
import argparse
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
import optuna
import sys
# file to be used by the optune library
# https://optuna.readthedocs.io/en/stable/reference/integration.html#optuna.integration.OptunaSearchCV


def main(trial,args):
    
    # Load config file
    load_cfg(cfg, args)

    # modify cfg file with optune parameters, suggest_int, suggest_float, suggest_categorical
    ## gnn
    cfg.gnn.layers_mp = trial.suggest_int("layers_mp", 1, 4)
    cfg.gnn.dim_inner = trial.suggest_int("dim_inner", 32, 512,step=32)
    cfg.gnn._layers_pre_mp = trial.suggest_categorical("layers_pre_mp",
                                                       [1, 2, 3, 4])
    cfg.gnn._layers_post_mp = trial.suggest_categorical(
        "layers_post_mp", [1, 2, 3, 4])
    cfg.gnn.layer_type = trial.suggest_categorical("layer_type", [
        "gcnconv", "gatconv", "ginconv", "sageconv", "generaledgeconv",
        "modgeneraledgeconv"
    ])
    cfg.gnn.dropout = trial.suggest_float("dropout", 0.1, 0.8, step=0.1)
    ## ae
    if args.dataset == "pathmnist":
        print("Using  - pathmnist")
        cfg.dataset.dir = trial.suggest_categorical("dataset.dir", [
            # "/home/uz1/projects/GCN/GraphGym/run/graph-data---pathmnist-32-128.h5",
            # "/home/uz1/projects/GCN/GraphGym/run/graph-data---pathmnist-32-256-UC_True.h5",
            # "/home/uz1/projects/GCN/GraphGym/run/graph-data---pathmnist-64-128.h5",
            # "/home/uz1/projects/GCN/GraphGym/run/graph-data---pathmnist-64-512-UC_True.h5",
            # "/home/uz1/graph-data---pathmnist-64.h5", # 64 - 256
            # "/home/uz1/projects/GCN/GraphGym/run/graph-data---pathmnist-128-256.h5",
            # "/home/uz1/projects/GCN/GraphGym/run/graph-data---pathmnist-128-512-UC_True.h5",

            # "/home/uz1/projects/GCN/GraphGym/run/graph-data---pathmnist-8-128-UC_False.h5",
            # "/home/uz1/projects/GCN/GraphGym/run/graph-data---pathmnist-16-128-UC_False.h5",
            # "/home/uz1/projects/GCN/GraphGym/run/graph-data---pathmnist-32-128-UC_False.h5",
            # "/home/uz1/projects/GCN/GraphGym/run/graph-data---pathmnist-64-128-UC_False.h5",

            "/home/uz1/graph-data---pathmnist-16-8-128-UC_False.h5",
            "/home/uz1/graph-data---pathmnist-16-32-128-UC_False.h5",
            "/home/uz1/graph-data---pathmnist-16-64-128-UC_False.h5",

        ])
    elif args.dataset == "bloodmnist":
        print("Using  - bloodmnist")
        cfg.dataset.dir = trial.suggest_categorical("dataset.dir", [
            "/home/uz1/projects/GCN/GraphGym/run/graph-data---bloodmnist-32-128.h5",
            "/home/uz1/projects/GCN/GraphGym/run/graph-data---bloodmnist-32-256.h5",
            "/home/uz1/projects/GCN/GraphGym/run/graph-data---bloodmnist-32-512.h5",
            "/home/uz1/projects/GCN/GraphGym/run/graph-data---bloodmnist-64-128.h5",
            "/home/uz1/projects/GCN/GraphGym/run/graph-data---bloodmnist-64-256.h5",
            "/home/uz1/projects/GCN/GraphGym/run/graph-data---bloodmnist-64-512.h5",
        ])
    cfg.dataset.limit = args.limit

    # out dir is based on dir name  file

    set_out_dir(
        cfg.dataset.dir.split('/')[-1] + "-" + cfg.out_dir, args.cfg_file)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)
    # Repeat for different random seeds
    for i in range(1):
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
        if cfg.train.mode == 'standard':
            accuracy = train(loggers, loaders, model, optimizer, scheduler,trial)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler)
    return accuracy


if __name__ == '__main__':
    
    sampler=optuna.samplers.NSGAIISampler(seed=42)

    # inin fake args
    parser = argparse.ArgumentParser(description=f"GraphGym + Optuna ")
    parser.add_argument('--dataset', type=str, default=None,required=True)
    parser.add_argument('--tag', type=str, default="",required=False)
    parser.add_argument('--limit', type=int, default=11000,required=False)
    args = parser.parse_args() 
    args.cfg_file = "/home/uz1/projects/GCN/GraphGym/run/configs/pyg/example_graph_cluster_copy.yaml"
    args.opts = []

    pruner = optuna.pruners.HyperbandPruner() if sampler is not optuna.samplers.NSGAIISampler() else None
    # study name is name + date
    study_name = args.dataset + " - " + str(sampler.__class__.__name__) + f"#{args.tag}" if args.tag != "" else args.dataset + " (" + datetime.datetime.now().strftime("%Y/%m/%d") + ")" + " - " + str(sampler.__class__.__name__)
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    
    #create optune study
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name=study_name,
        pruner=pruner,
        load_if_exists=True,
        sampler=sampler ) # Specify the study name here.
    # Run training
    study.optimize(
        lambda trial: main(trial,args),
        n_trials=100,
    )
