import sys
sys.path.append('Example2')

from Example2.GNN import *
import copy
import os 

if 'IS_GRADESCOPE_ENV' not in os.environ:

    # Define arguments
    args = { 
        'device': 'cpu',
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