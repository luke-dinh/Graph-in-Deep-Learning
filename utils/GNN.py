import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, 
                dropout, return_embeds=False):

        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.return_embeds = return_embeds
        self.dropout = dropout

        # Define the convs
        self.convs = nn.ModuleList([GCNConv(input_dim, hidden_dim)])
        self.convs.extend(nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for i in range(num_layers-2)]))
        self.convs.extend(nn.ModuleList([GCNConv(hidden_dim, output_dim)]))

        # Define the batchnorm
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for i in range(num_layers-1)])

        # Define the log max 
        self.log_max = nn.LogSoftmax(dim=-1)

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

        out = x if self.return_embeds else self.log_max(x)

        return out

class train_and_eval():

    def train(model, data, train_idx, optimizer, loss_fn):

        # Set the model to train mode
        model.train()

        # Zero grad the optimizer
        optimizer.zero_grad()

        # Feed the data into the model
        out = model(data.x, data.adj_t)

        # Load the output to loss funtion
        loss = loss_fn(out[train_idx], data.y[train_idx].squeeze(1))

        # Update loss and optimizer
        loss.backward()
        optimizer.step()

        return loss.item()

    def eval(model, data, split_idx, evaluator, save_model_results=False):

        with torch.no_grad():

            # Set the model to eval mode
            model.eval()

            # Load the data to model
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


            return train_acc, valid_acc, test_acc