
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

def main(trial,args):
    
    #withc to suggest categorical of only two ints 28 and 224 python projects\GraphGym\full_pipe.py --dataset dermamnist --tag "parallel-4"
    image_size = 224
    patch_size = trial.suggest_int("patch_size", 8, 112, step=8)
    num_patches = (image_size // patch_size) ** 2
    if num_patches in  [0,1,2,3]: 
        print("Too little patches per image !")
        raise optuna.exceptions.TrialPruned()
    
    num_nodes = trial.suggest_int("num_nodes", 4, 224, step=4)
    # numer of nodes should be less than the maximum number of patches

    dataset = args.dataset
    # batch size should depend oon num_patches by 
    batch_size = 128 // (num_patches)
    if batch_size == 0:
        batch_size = 1
    # check if trial already exists - running or complete or pruned  - if failed it should run again
    for previous_trial in trial.study.trials[:-1]:
         if (previous_trial.state == TrialState.RUNNING ) and trial.params == previous_trial.params:
             print(f"Duplicated trial already Ran: {trial.params}")
             raise optuna.exceptions.TrialPruned()
    if num_patches <10:
        batch_size = batch_size // 2
    if patch_size > 80 and num_patches > 10 :
        batch_size = batch_size // 2
    print(f"Trial run with \n image_size: {image_size}, patch_size: {patch_size}, num_nodes: {num_nodes}, batch_size: {batch_size}")
    # run vae_pipe.py 
    import subprocess
    subprocess.run(["python", r"C:\Users\Usama\projects\GCNs\vae_pipe.py", "--img_size", str(image_size), "--patch_size", str(patch_size), "--batch_size", str(batch_size),"-use_pretrain","True","-k",str(num_nodes),"--dataset", str(dataset)])
    # run gen_graph.py
    out = subprocess.run(["python", r"C:\Users\Usama\projects\GCNs\gen_graphs.py", "-size", str(image_size), "--dataset", str(dataset),"--k",str(num_nodes),"--batch_size", str(batch_size),"--patch_size", str(patch_size)],capture_output=True)

    # check out code
    print(out.stdout.decode("utf-8"))
    if "File already exists" in out.stdout.decode("utf-8"):
        print("Trial exists")
        raise optuna.exceptions.TrialPruned()

    # run graphgym - on three different cfgs with graph generated above 
    g_data_path = rf"C:\Users\Usama\data\graph-data---{dataset}-{patch_size}-{num_nodes}-{image_size}-UC_False.h5" 
    MAIN = r"C:\Users\Usama\projects\GraphGym\run\main.py"
    args.cfg_file = r"C:\Users\Usama\projects\GraphGym\run\configs\pyg\example_graph_cluster_copy.yaml"

    load_cfg(cfg, args)
    cfg.dataset.dir = g_data_path
    # run graphgym on cf
    cfg.gnn.layer_type = "modgeneraledgeconv"
    cfg.share.dim_out= 9 if dataset == "pathmnist" else 7
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
    acc1 = train(loggers, loaders, model, optimizer, scheduler, trial)

    # get accuracy average - make sure they're all floats
    return acc1



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
    sampler=optuna.samplers.TPESampler(seed=44)

    # inin fake args
    parser = argparse.ArgumentParser(description=f"fullpipline Optuna ")
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
    n_trials=200,
    )
