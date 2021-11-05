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

num_classes = get_num_classes(pyg_dataset)
num_features = get_num_features(pyg_dataset)

print("Number of classes: {}".format(num_classes))
print("Number of features: {}".format(num_features))