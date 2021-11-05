import os 
import torch
from torch_geometric.datasets import TUDataset

# Check torch version (to install torch_geometric: torch >=1.8.0)
# print(torch.__version__)

#  Get the dataset

if 'IS_GRADESCOPE_ENV' not in os.environ:
    root = './enzymes'
    name = 'ENZYMES'

    # The ENZYMES dataset
    pyg_dataset = TUDataset(root, name)

    # 600 graphs in this dataset
    print(pyg_dataset)

# Analyzing the dataset

def get_num_classes(pyg_dataset):

    num_classes = 0
    num_classes += pyg_dataset.num_classes

    return num_classes

def get_num_features(pyg_dataset):

    num_features = 0
    num_features += pyg_dataset.num_features

    return num_features

# Get the label of the graph with index 100 

def get_graph_class(pyg_dataset, idx):

    label = pyg_dataset[idx].y

    return label 

# Analyzing the graph with index i in dataset

def get_graph_num_edges(pyg_dataset, idx):

    num_edges = pyg_dataset[idx].num_edges

    return num_edges
