import sys
sys.path.append('Example2')

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from Example2.GNN import *
import copy
import os 

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

    # Define arguments
    args = { 
        'device': device,
        'num_layers': 3,
        'hidden_dim': 256,
        'dropout': 0.5,
        'lr': 0.01,
        'epochs': 50
    }

    # Load the model
    model = GCN( 
        data.num_features,
        args['hidden_dim'],
        dataset.num_classes,
        args['num_layers'],
        args['dropout']
    ).to(args['device'])

    # Load the evaluator
    evaluator = Evaluator(name='ogbn-arxiv')