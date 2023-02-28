
from graphgym.config import cfg, dump_cfg, load_cfg, set_run_dir, set_out_dir

from graphgym.cmd_args import parse_args
from graphgym.config import cfg, dump_cfg, load_cfg, set_run_dir, set_out_dir
from graphgym.loader import create_dataset, create_loader
from graphgym.logger import create_logger, setup_printing
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

def main(trial,args):
    
    image_size = trial.suggest_int("image_size", 28, 512, step=4)
    patch_size = trial.suggest_int("patch_size", 8, 104, step=8)
    num_patches = (image_size // patch_size) ** 2
    if num_patches == 0: 
        print("Too many patches per image !")
        raise optuna.exceptions.TrialPruned()
    
    num_nodes = trial.suggest_int("num_nodes", 4, 128, step=4)
    dataset = "pathmnist"
    # batch size should depend oon num_patches by 
    batch_size = 128 // (num_patches)
    if batch_size == 0:
        print("Too many patches per image !")
        raise optuna.exceptions.TrialPruned()
    # check if trial already exists - running or complete or pruned  - if failed it should run again
    for previous_trial in trial.study.trials[:-1]:
        if (previous_trial.state == TrialState.RUNNING or previous_trial.state == TrialState.COMPLETE  or previous_trial.state == TrialState.PRUNED) and trial.params == previous_trial.params:
            print(f"Duplicated trial: {trial.params}")
            raise optuna.exceptions.TrialPruned()
    # if num_patches <10:
    #     batch_size = 16
    print(f"image_size: {image_size}, patch_size: {patch_size}, num_nodes: {num_nodes}, batch_size: {batch_size}")
    # run vae_pipe.py 
    import subprocess
    subprocess.run(["python", "/home/uz1/projects/GCN/vae_pipe.py", "--img_size", str(image_size), "--patch_size", str(patch_size), "--batch_size", str(batch_size),"-use_pretrain","True","-k",str(num_nodes)])
    # run gen_graph.py
    out = subprocess.run(["python", "/home/uz1/projects/GCN/gen_graphs.py", "-size", str(image_size), "--dataset", str(dataset),"--k",str(num_nodes),"--batch_size", str(batch_size),"--patch_size", str(patch_size)],capture_output=True)

    # check out code
    print(out.stdout.decode("utf-8"))
    if "File already exists" in out.stdout.decode("utf-8"):
        print("Trial exists")
        raise optuna.exceptions.TrialPruned()

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
    acc1 = train(loggers, loaders, model, optimizer, scheduler)

    # run graphgym on cfg2
    cfg.gnn.layer_type = "sageconv"
    model = create_model()
    optimizer = create_optimizer(model.parameters())
    scheduler = create_scheduler(optimizer)
    # Print model info
    logging.info(model)
    logging.info(cfg)
    cfg.params = params_count(model)
    logging.info('Num parameters: %s', cfg.params)
    # Start training
    acc2 = train(loggers, loaders, model, optimizer, scheduler)
    
    # run graphgym on 
    cfg.gnn.layer_type = "gatconv"
    model = create_model()
    optimizer = create_optimizer(model.parameters())
    scheduler = create_scheduler(optimizer)
    # Print model info
    logging.info(model)
    logging.info(cfg)
    cfg.params = params_count(model)
    logging.info('Num parameters: %s', cfg.params)
    # Start training
    acc3 = train(loggers, loaders, model, optimizer, scheduler)
    
    # get accuracy average - make sure they're all floats
    acc = (float(acc1) + float(acc2) + float(acc3)) / 3
    return acc



if __name__ == "__main__":
    import optuna
    import argparse
    import datetime
    import logging
    import sys
    import os
    import subprocess
    import torch

    # set seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # set sampler
    sampler=optuna.samplers.NSGAIISampler(seed=42)

        # inin fake args
    parser = argparse.ArgumentParser(description=f"GraphGym + Optuna ")
    parser.add_argument('--dataset', type=str, default=None,required=True)
    parser.add_argument('--tag', type=str, default="",required=False)
    parser.add_argument('--limit', type=int, default=11000,required=False)
    args = parser.parse_args() 
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
