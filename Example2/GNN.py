import os 

import torch
import pandas as pd
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.nn import GCNConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


if 'IS_GRADESCOPE_ENV' not in os.environ:

    # Load the dataset
    dataset_name = 'ogbn-arxiv'
    dataset = PygNodePropPredDataset(name=dataset_name,
                                  transform=T.ToSparseTensor())
    data = dataset[0]

    # Make the adjacency matrix to symmetric
    data.adj_t = data.adj_t.to_symmetric()

    # Only cpu is valilable
    device = 'cpu'

    # Load the data to device
    data = data.to(device)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)