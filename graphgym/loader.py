import logging
import pickle
import time

import networkx as nx
from graphgym.datasets import ImageTOGraphDataset, ImageToClusterHD5
import torch
import torch_geometric.transforms as T
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch.utils.data import DataLoader
from torch_geometric.datasets import (PPI, Amazon, Coauthor, KarateClub,
                                      MNISTSuperpixels, Planetoid, QM7b,
                                      TUDataset)

import graphgym.models.feature_augment as preprocess
import graphgym.register as register
from graphgym.config import cfg
from graphgym.models.transform import (edge_nets, ego_nets, path_len,
                                       remove_node_feature)
from graphgym.datasets import VAE, ImgToGraph, medmnist_modified,ConcatDataset,DivideIntoPatches

from medmnist.dataset import PathMNIST, BreastMNIST,OCTMNIST,ChestMNIST,PneumoniaMNIST,DermaMNIST,RetinaMNIST,BloodMNIST,TissueMNIST,OrganAMNIST,OrganCMNIST,OrganSMNIST
def load_pyg(name, dataset_dir,limit=None):
    '''
    load pyg format dataset
    :param name: dataset name
    :param dataset_dir: data directory
    :return: a list of networkx/deepsnap graphs
    '''
    # dataset_dir = '{}/{}'.format(dataset_dir, name)
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset_raw = Planetoid(dataset_dir, name)
    elif name[:3] == 'TU_':
        # TU_IMDB doesn't have node features
        if name[3:] == 'IMDB':
            name = 'IMDB-MULTI'
            dataset_raw = TUDataset(dataset_dir, name, transform=T.Constant())
        else:
            dataset_raw = TUDataset(dataset_dir, name[3:])
        # TU_dataset only has graph-level label
        # The goal is to have synthetic tasks
        # that select smallest 100 graphs that have more than 200 edges
        if cfg.dataset.tu_simple and cfg.dataset.task != 'graph':
            size = []
            for data in dataset_raw:
                edge_num = data.edge_index.shape[1]
                edge_num = 9999 if edge_num < 200 else edge_num
                size.append(edge_num)
            size = torch.tensor(size)
            order = torch.argsort(size)[:100]
            dataset_raw = dataset_raw[order]
    elif name == 'Karate':
        dataset_raw = KarateClub()
    elif 'Coauthor' in name:
        if 'CS' in name:
            dataset_raw = Coauthor(dataset_dir, name='CS')
        else:
            dataset_raw = Coauthor(dataset_dir, name='Physics')
    elif 'Amazon' in name:
        if 'Computers' in name:
            dataset_raw = Amazon(dataset_dir, name='Computers')
        else:
            dataset_raw = Amazon(dataset_dir, name='Photo')
    elif name == 'MNIST':
        dataset_raw = MNISTSuperpixels(dataset_dir)
    elif name == 'PPI':
        dataset_raw = PPI(dataset_dir)
    elif name == 'QM7b':
        dataset_raw = QM7b(dataset_dir)
    elif name == 'medmnist-path':
        dataset_raw = medmnist_modified(root=dataset_dir,split="train",download=True,flag="pathmnist",num_classes=9)
    elif name == 'medmnist-path-cluster':
        from torchvision import transforms
       
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.ConvertImageDtype(torch.float),
            DivideIntoPatches(patch_size=32), # takes an image tensor and returns a list of patches stacked as (H // patch_size **2 x H x W x C)
        ])
        vae = VAE(input_height=32, latent_dim=256)
        vae = vae.load_from_checkpoint("/home/uz1/projects/GCN/logging/PathMNIST/epoch=7-step=89992.ckpt")

        data = PathMNIST(root='/home/uz1/DATA!/medmnist', download=True,split='train',transform=transform)

        dataset_raw = ImageTOGraphDataset(data=data,vae=vae,kmeans="/home/uz1/projects/GCN/kmeans-model-128-32-8-PathMNIST.pkl")
    elif "cluster" in name:
        # get nc from path : /home/uz1/projects/GCN/GraphGym/run/graph-data---pathmnist-64-128.h5 -> 64
        nc = int(dataset_dir.split("-")[-2]) if len(dataset_dir.split("-"))>2 else None
        # check if nc is in [16,32,64,128]
        nc = nc if nc in [16,32,64,128] else None
        dataset_raw = ImageToClusterHD5(data=dataset_dir,n_clusters=nc)#,limit=51000)
    elif name =="retinamnist":
        dataset_train = medmnist_modified(root=dataset_dir,split="train",download=True,flag="retinamnist")
        dataset_val = medmnist_modified(root=dataset_dir,split="val",download=True,flag="retinamnist")
        dataset_test = medmnist_modified(root=dataset_dir,split="test",download=True,flag="retinamnist")
        dataset_raw = ConcatDataset([dataset_train,dataset_val,dataset_test])
    elif name =="breastmnist":
        dataset_train = medmnist_modified(root=dataset_dir,split="train",download=True,flag="breastmnist")
        dataset_val = medmnist_modified(root=dataset_dir,split="val",download=True,flag="breastmnist")
        dataset_test = medmnist_modified(root=dataset_dir,split="test",download=True,flag="breastmnist")
        dataset_raw = ConcatDataset([dataset_train,dataset_val,dataset_test])
    elif name =="chestmnist":
        dataset_train = medmnist_modified(root=dataset_dir,split="train",download=True,flag="chestmnist")
        dataset_val = medmnist_modified(root=dataset_dir,split="val",download=True,flag="chestmnist")
        dataset_test = medmnist_modified(root=dataset_dir,split="test",download=True,flag="chestmnist")
        dataset_raw = ConcatDataset([dataset_train,dataset_val,dataset_test])
    elif name =="dermamnist":
        dataset_train = medmnist_modified(root=dataset_dir,split="train",download=True,flag="dermamnist")
        dataset_val = medmnist_modified(root=dataset_dir,split="val",download=True,flag="dermamnist")
        dataset_test = medmnist_modified(root=dataset_dir,split="test",download=True,flag="dermamnist")
        dataset_raw = ConcatDataset([dataset_train,dataset_val,dataset_test])
    elif name == "bloodmnist":
        dataset_train = medmnist_modified(root=dataset_dir,split="train",download=True,flag="bloodmnist")
        dataset_val = medmnist_modified(root=dataset_dir,split="val",download=True,flag="bloodmnist")
        dataset_test = medmnist_modified(root=dataset_dir,split="test",download=True,flag="bloodmnist")
        dataset_raw = ConcatDataset([dataset_train,dataset_val,dataset_test])
    elif name == "octmnist":
        dataset_train = medmnist_modified(root=dataset_dir,split="train",download=True,flag="octmnist")
        dataset_val = medmnist_modified(root=dataset_dir,split="val",download=True,flag="octmnist")
        dataset_test = medmnist_modified(root=dataset_dir,split="test",download=True,flag="octmnist")
        dataset_raw = ConcatDataset([dataset_train,dataset_val,dataset_test])
    elif name =="pathmnist":
        dataset_train = medmnist_modified(root=dataset_dir,split="train",download=True,flag="pathmnist")
        dataset_val = medmnist_modified(root=dataset_dir,split="val",download=True,flag="pathmnist")
        dataset_test = medmnist_modified(root=dataset_dir,split="test",download=True,flag="pathmnist")
        dataset_raw = ConcatDataset([dataset_train,dataset_val,dataset_test])
    else:
        raise ValueError('{} not support'.format(name))
    graphs = GraphDataset.pyg_to_graphs(dataset_raw)
    return graphs


def load_nx(name, dataset_dir):
    '''
    load networkx format dataset
    :param name: dataset name
    :param dataset_dir: data directory
    :return: a list of networkx/deepsnap graphs
    '''
    try:
        with open('{}/{}.pkl'.format(dataset_dir, name), 'rb') as file:
            graphs = pickle.load(file)
    except Exception:
        graphs = nx.read_gpickle('{}/{}.gpickle'.format(dataset_dir, name))
        if not isinstance(graphs, list):
            graphs = [graphs]
    return graphs


def load_dataset():
    '''
    load raw datasets.
    :return: a list of networkx/deepsnap graphs, plus additional info if needed
    '''
    format = cfg.dataset.format
    name = cfg.dataset.name
    # dataset_dir = '{}/{}'.format(cfg.dataset.dir, name)
    dataset_dir = cfg.dataset.dir
    # Try to load customized data format
    for func in register.loader_dict.values():
        graphs = func(format, name, dataset_dir)
        if graphs is not None:
            return graphs
    # Load from Pytorch Geometric dataset
    if format == 'PyG':
        graphs = load_pyg(name, dataset_dir,cfg.dataset.limit if hasattr(cfg.dataset, "limit") else None)
    # Load from networkx formatted data
    # todo: clean nx dataloader
    elif format == 'nx':
        graphs = load_nx(name, dataset_dir)
    # Load from OGB formatted data
    elif cfg.dataset.format == 'OGB':
        if cfg.dataset.name == 'ogbg-molhiv':
            dataset = PygGraphPropPredDataset(name=cfg.dataset.name)
            graphs = GraphDataset.pyg_to_graphs(dataset)
        # Note this is only used for custom splits from OGB
        split_idx = dataset.get_idx_split()
        return graphs, split_idx
    else:
        raise ValueError('Unknown data format: {}'.format(cfg.dataset.format))
    return graphs


def filter_graphs():
    '''
    Filter graphs by the min number of nodes
    :return: min number of nodes
    '''
    if cfg.dataset.task == 'graph':
        min_node = 0
    else:
        min_node = 5
    return min_node


def transform_before_split(dataset):
    '''
    Dataset transformation before train/val/test split
    :param dataset: A DeepSNAP dataset object
    :return: A transformed DeepSNAP dataset object
    '''
    if cfg.dataset.remove_feature:
        dataset.apply_transform(remove_node_feature,
                                update_graph=True,
                                update_tensor=False)
    augmentation = preprocess.FeatureAugment()
    actual_feat_dims, actual_label_dim = augmentation.augment(dataset)
    if cfg.dataset.augment_label:
        dataset.apply_transform(preprocess._replace_label,
                                update_graph=True,
                                update_tensor=False)
    # Update augmented feature/label dims by real dims (user specified dims
    # may not be realized)
    cfg.dataset.augment_feature_dims = actual_feat_dims
    if cfg.dataset.augment_label:
        cfg.dataset.augment_label_dims = actual_label_dim

    # Temporary for ID-GNN path prediction task
    if cfg.dataset.task == 'edge' and 'id' in cfg.gnn.layer_type:
        dataset.apply_transform(path_len,
                                update_graph=False,
                                update_tensor=False)

    return dataset


def transform_after_split(datasets):
    '''
    Dataset transformation after train/val/test split
    :param dataset: A list of DeepSNAP dataset objects
    :return: A list of transformed DeepSNAP dataset objects
    '''
    if cfg.dataset.transform == 'ego':
        for split_dataset in datasets:
            split_dataset.apply_transform(ego_nets,
                                          radius=cfg.gnn.layers_mp,
                                          update_tensor=True,
                                          update_graph=False)
    elif cfg.dataset.transform == 'edge':
        for split_dataset in datasets:
            split_dataset.apply_transform(edge_nets,
                                          update_tensor=True,
                                          update_graph=False)
            split_dataset.task = 'node'
        cfg.dataset.task = 'node'
    return datasets


def set_dataset_info(datasets):
    r"""
    Set global dataset information

    Args:
        datasets: List of dataset object

    """
    # get dim_in and dim_out
    try:
        cfg.share.dim_in = datasets[0].num_node_features
    except Exception:
        cfg.share.dim_in = 1
    try:
        cfg.share.dim_out = datasets[0].num_labels
        if 'classification' in cfg.dataset.task_type and \
                cfg.share.dim_out == 2:
            cfg.share.dim_out = 1
    except Exception:
        cfg.share.dim_out = 1

    # count number of dataset splits
    cfg.share.num_splits = len(datasets)


def create_dataset():
    # Load dataset
    time1 = time.time()
    if cfg.dataset.format == 'OGB':
        graphs, splits = load_dataset()
    else:
        graphs = load_dataset()
    # check if graphs is none
    if graphs is []:
        raise ValueError('Graphs is None')
    # Filter graphs
    time2 = time.time()
    min_node = filter_graphs()

    # Create whole dataset
    dataset = GraphDataset(
        graphs,
        task=cfg.dataset.task,
        edge_train_mode=cfg.dataset.edge_train_mode,
        edge_message_ratio=cfg.dataset.edge_message_ratio,
        edge_negative_sampling_ratio=cfg.dataset.edge_negative_sampling_ratio,
        resample_disjoint=cfg.dataset.resample_disjoint,
        minimum_node_per_graph=min_node)

    # Transform the whole dataset
    dataset = transform_before_split(dataset)

    # Split dataset
    time3 = time.time()
    # Use custom data splits
    if cfg.dataset.format == 'OGB':
        datasets = []
        datasets.append(dataset[splits['train']])
        datasets.append(dataset[splits['valid']])
        datasets.append(dataset[splits['test']])
    # Use random split, supported by DeepSNAP
    else:
        datasets = dataset.split(transductive=cfg.dataset.transductive,
                                 split_ratio=cfg.dataset.split,
                                 shuffle=cfg.dataset.shuffle_split)
    # We only change the training negative sampling ratio
    for i in range(1, len(datasets)):
        dataset.edge_negative_sampling_ratio = 1

    # Transform each split dataset
    time4 = time.time()
    datasets = transform_after_split(datasets)
    set_dataset_info(datasets)

    time5 = time.time()
    logging.info('Load: {:.4}s, Before split: {:.4}s, '
                 'Split: {:.4}s, After split: {:.4}s'.format(
                     time2 - time1, time3 - time2, time4 - time3,
                     time5 - time4))

    return datasets


def create_loader(datasets):
    loader_train = DataLoader(datasets[0],
                              collate_fn=Batch.collate(),
                              batch_size=cfg.train.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers,
                              pin_memory=False)

    loaders = [loader_train]
    for i in range(1, len(datasets)):
        loaders.append(
            DataLoader(datasets[i],
                       collate_fn=Batch.collate(),
                       batch_size=cfg.train.batch_size,
                       shuffle=False,
                       num_workers=cfg.num_workers,
                       pin_memory=False))

    return loaders
