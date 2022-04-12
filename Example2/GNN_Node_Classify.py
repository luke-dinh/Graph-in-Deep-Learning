import torch
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, 
                    dropout, return_embeds = False):

        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.return_embeds = return_embeds

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
    def forward(self, x, adj_t):

        for conv, bn in zip(self.convs[:-1], self.batch_norm):
            x1 = F.relu(bn(conv(x, adj_t)))
            if self.training:
                x1 = F.dropout(x1, p=self.dropout)
            x = x1
        x = self.convs[-1](x, adj_t)
        out = x if self.return_embeds else self.softmax(x)

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

@torch.no_grad()

def test(model, data, split_idx, evaluator, save_model_results=False):

    # Load the model
    model.eval()

    # Define the output of the model
    out = model(data.x, data.adj_t)

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    if save_model_results:
      print ("Saving Model Predictions")

      data = {}
      data['y_pred'] = y_pred.view(-1).cpu().detach().numpy()

      df = pd.DataFrame(data=data)
      # Save locally as csv
      df.to_csv('ogbn-arxiv_node.csv', sep=',', index=False)


    return train_acc, valid_acc, test_acc, out 