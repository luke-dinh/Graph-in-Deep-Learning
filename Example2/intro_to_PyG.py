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