import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch_geometric.nn.conv import MessagePassing


class GraphSage(MessagePassing):

    def __init__(self, in_channels, out_channels, normalize=True, bias=False, **kwargs):

        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        # Layers needed for the message
        self.lin_l = nn.Linear(self.in_channels, self.out_channels)
        self.lin_r = nn.Linear(self.in_channels, self.out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size=None):

        prop = self.propagate(edge_index, x=(x,x), size=size)
        out = self.lin_l(x) + self.lin_r(prop)

        if self.normalize:
            out = F.normalize(out, p=2)

        return out

    def message(self, x_j):

        out = x_j
        return x_j

    def aggregate(self, inputs, index, dim_size=None):

        node_dim = self.node_dim
        out = torch_scatter.scatter(inputs, index, node_dim, dim_size=dim_size, reduce='mean')

        return out