import sys

from torch.utils.data.dataset import Dataset
sys.path.append('.')

import os
import copy
from tqdm.notebook import tqdm

import torch.nn as nn
from Example2.GNN_Graph_Classification import *
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader 

if 'IS_GRADESCOPE_ENV' not in os.environ:

    # Load the dataset
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv')

    device = 'cpu'
    split_idx = dataset.get_idx_split()

    train_loader = DataLoader(dataset[split_idx['train']], batch_size=32, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset[split_idx['test']], batch_size=32, shuffle=False, num_workers=0)

    # Args
    args = { 
        'device': device,
        'num_layers': 5,
        'hidden_dim': 256,
        'dropout': 0.5,
        'lr': 0.001,
        'epochs': 30
    }

    # Load the model
    model = GCN_Graph(args['hidden_dim'], dataset.num_tasks, args['num_layers'],
                        args['dropout']).to(device)

    evaluator = Evaluator(name='ogbg-molhiv')

    # Reset parameters
    model.reset_parameters()

    # Loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters, lr=args['lr'])

    # Training
    best_model = None
    best_valid_acc = 0

    for epoch in range(args['epochs']):

        print('Training...')
        loss = train(model, device, train_loader, optimizer, loss_fn)

        print('Evaluating...')
        train_result = eval(model, device, train_loader, evaluator)
        valid_result = eval(model, device, valid_loader, evaluator)
        test_result = eval(model, device, test_loader, evaluator)

        train_acc = train_result[dataset.eval_metric]
        valid_acc = valid_result[dataset.eval_metric]
        test_acc = test_result[dataset.eval_metric]

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc 
            best_model = copy.deepcopy(model)

        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}% '
              f'Test: {100 * test_acc:.2f}%')

        # Get the best result
        train_acc = eval(best_model, device, train_loader, evaluator)[dataset.eval_metric]
        valid_acc = eval(best_model, device, valid_loader, evaluator, save_model_results=True, save_file="valid")[dataset.eval_metric]
        test_acc  = eval(best_model, device, test_loader, evaluator, save_model_results=True, save_file="test")[dataset.eval_metric]

        print(f'Best model: '
            f'Train: {100 * train_acc:.2f}%, '
            f'Valid: {100 * valid_acc:.2f}% '
            f'Test: {100 * test_acc:.2f}%')