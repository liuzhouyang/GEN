from torch_geometric.utils import to_networkx
from torch_geometric.datasets import GEDDataset
import networkx as nx
#from NodeMap import NodeMapDataset
import random
import os.path as osp
import torch

def get_syndataset(args):
    dataset = torch.load(f'../datasetStorage/GEN/min_node=100_max_node=400.pt')
    random.shuffle(dataset)
    train_size = int(len(dataset) * (1 - args.test_pct))
    trainset, testset = dataset[:train_size], dataset[train_size:]
    train_size = int(len(trainset) * (1 - args.val_pct))
    trainset, valset = trainset[:train_size], trainset[train_size:]
    return trainset, valset, testset

def get_dataset(args):
    num_node_labels, graphs, trainset, valset = load_ged_dataset(args, train = True)
    _, graphs_test, testset = load_ged_dataset(args, train = False)
    return num_node_labels, (graphs, graphs_test), (trainset, valset, testset)

def get_mixed_ged(args):
    train_path = torch.load(f'../datasetStorage/di4edit/dataset/graphs/{args.dataset_name.upper()}/processed/{args.dataset_name.upper()}_gt_12131_train.pt') + torch.load(f'../datasetStorage/di4edit/dataset/graphs/{args.dataset_name.upper()}/processed/{args.dataset_name.upper()}_gt_32302_train.pt')
    test_path = torch.load(f'../datasetStorage/di4edit/dataset/graphs/{args.dataset_name.upper()}/processed/{args.dataset_name.upper()}_gt_12131_test.pt') + torch.load(f'../datasetStorage/di4edit/dataset/graphs/{args.dataset_name.upper()}/processed/{args.dataset_name.upper()}_gt_32302_test.pt')
    random.shuffle(train_path)
    random.shuffle(test_path)   
    train_size = int(len(train_path) * (1 - args.val_pct))
    train_path, val_path = train_path[:train_size], train_path[train_size:]
    return train_path, val_path, test_path

def get_ged(args):
    train_path = torch.load(f'../datasetStorage/di4edit/dataset/graphs/{args.dataset_name.upper()}/processed/{args.dataset_name.upper()}_gt_{args.costs[args.cost_setting]}_train.pt')
    test_path = torch.load(f'../datasetStorage/di4edit/dataset/graphs/{args.dataset_name.upper()}/processed/{args.dataset_name.upper()}_gt_{args.costs[args.cost_setting]}_test.pt')
    random.shuffle(train_path)
    random.shuffle(test_path)   
    train_size = int(len(train_path) * (1 - args.val_pct))
    train_path, val_path = train_path[:train_size], train_path[train_size:]
    return train_path, val_path, test_path

def load_ged_dataset(args, train = True):
    if args.dataset_name == "aids":
        dataset = GEDDataset(root="./GED/AIDS700nef",
                            name="AIDS700nef", train = train)
    elif args.dataset_name == "linux":
        dataset = GEDDataset(root="./GED/LINUX",
                            name="LINUX", train = train)
    elif args.dataset_name == "imdb":
        dataset = GEDDataset(root="./GED/IMDBMulti",
                            name="IMDBMulti", train = train)
    if dataset[0].x == None:
        num_node_labels = 0
    else:
        num_node_labels = dataset[0].x.size(-1)
    graphs = []
    for i, data in enumerate(dataset):
        if not type(data) == nx.Graph:
            if num_node_labels == 0:
                graph = to_networkx(data).to_undirected()
            else:
                graph = to_networkx(data, node_attrs = ['x']).to_undirected()
        graphs.append(graph)
    if train:
        train_len = int(len(dataset) * (1.0 - args.val_pct))
        return num_node_labels, graphs, dataset[:train_len], dataset[train_len:]
    else:
        return num_node_labels, graphs, dataset
