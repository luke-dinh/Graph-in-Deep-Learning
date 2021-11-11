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

        