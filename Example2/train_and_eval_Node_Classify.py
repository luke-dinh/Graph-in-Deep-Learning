import os
import copy
import sys
sys.path.append('.')

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
from Example2.GNN_Node_Classify import *

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

      # reset the parameters to initial random value
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    loss_fn = F.nll_loss

    best_model = None
    best_valid_acc = 0

    for epoch in range(1, 1 + args["epochs"]):
        loss = train(model, data, train_idx, optimizer, loss_fn)
        result = test(model, data, split_idx, evaluator)
        train_acc, valid_acc, test_acc = result
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(model)
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}% '
              f'Test: {100 * test_acc:.2f}%')