import os 

import torch
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn

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

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, 
                    dropout, return_embeds = False):

        super(GCN, self).__init__()

        # GCN_Convs layers
        self.convs = nn.ModuleList([GCNConv(input_dim, hidden_dim)])
        self.convs.extend(nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for i in range(num_layers-2)]))
        self.convs.extend(nn.ModuleList([GCNConv(hidden_dim, output_dim)]))

        # Batchnorm layers
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for i in range(num_layers-1)])

        # Log Softmax layers
        self.softmax = nn.LogSoftmax(dim=1)

        # Probability of an element getting zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norm:
            bn.reset_parameters()

    # Feed forward
    def forward(self, adj_t, x):

        for i in range(self.num_layers):

            # Last conv layers in the figure
            if (i == self.num_layers - 1):
                x = self.convs[i](x, adj_t)
                if (self.return_embeds):
                    return x
                out = self.softmax(x)
            
            # Other layers in the network
            else:
                x = self.convs[i](x, adj_t)
                x = self.batch_norm[i](x)
                x = F.relu(x)
                x = F.dropout(x, p = self.dropout, training=self.training)

        return out 

def train(model, data, train_idx, optimizer, loss_fn):

    model.train()

    # Zero grad the optimizer
    optimizer.zero_grad()

    # Feed the data into the model
    out = model(data.x, data.adj_t)

    # Slice the model output and label by train_idx and feed the sliced output into loss function
    loss = loss_fn(out[train_idx], data.y[train_idx].squeeze(1))

    # Update params
    loss.backward()
    optimizer.step()

    return loss.item()