import torch
import torch.nn as nn
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