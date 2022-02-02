import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)

from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class GNN_stack(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, args, emb=False):

        super(GNN_stack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))

        assert (args.num_layers >= 1), 'Number of layers is not >=1'

        for l in range(args.num_layers -1 ):
            self.convs.append(conv_model(args.heads * hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential( 
            nn.Linear(args.heads * hidden_dim, hidden_dim), nn.Dropout(args.dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.dropout = args.dropout
        self.num_layers = args.num_layers

        self.emb = emb

    def build_conv_model(self, model_type):
        if model_type == 'GraphSage':
            return GraphSage
        
        elif model_type == 'GAT':
            return GAT 

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Feed Forward
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.post_mp(x)

        if self.emb == True:
            return x

        return F.log_softmax(x, dim=1)

    # Loss
    def loss(self, pred, label):
        return F.nll_loss(pred, label)

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

class GAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=2, 
                    negative_slope=0.2, dropout=0., **kwargs):

        super(GAT, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        # Layers needed for the message functions
        self.lin_l = nn.Linear(self.in_channels, self.out_channels, self.heads)
        self.lin_r = self.lin_l

        # Define the attention parameters \overrightarrow{a_l/r}^T
        self.att_l = nn.Parameter(torch.zeros(self.heads, self.out_channels))
        self.att_r = nn.Parameter(torch.zeros(self.heads, self.out_channels))

        self.reset_parameters_2()

        # Reset parameters
    def reset_parameters_2(self):

        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

        