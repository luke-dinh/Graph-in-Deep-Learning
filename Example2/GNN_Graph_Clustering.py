import tqdm
import sys
sys.path.append('.')
from Example2.GNN_Node_Classify import GCN

import torch
import torch.nn as nn

from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_add_pool, global_mean_pool

class GCN_Graph(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_Graph, self).__init__()

        # Load encoders for Atoms in molecule graphs
        self.node_encoder = AtomEncoder(hidden_dim)

        # Node embedding model
        self.gnn_node = GCN(hidden_dim, hidden_dim, hidden_dim, 
                            num_layers, dropout, return_embeds=True)

        # Initialize pooling layer as a global mean pooling layer
        self.pool_layer = global_mean_pool()

        # Output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def reste_parameters(self):
        self.gnn_node.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, batched_data):

        # Extract important attribute of the graph
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        embedded = self.node_encoder(x)

        # Forward feed
        out = self.gnn_node(embedded, edge_index)
        out = self.pool_layer(out, batch)
        out = self.linear(out)

        return out

def train(model, device, data_loader, optimizer, loss_fn):

    model.train()
    loss = 0

    for step, batch in enumerate(tqdm(data_loader, desc='Iteration')):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            ## Ignore nan values
            is_labeled = batch.y == batch.y

            # Zero grad optimizer
            optimizer.zero_grad()

            # Feed the data into the model
            out = model(batch)

            ## 3. Use `is_labeled` mask to filter output and labels
            ## 4. You may need to change the type of label to torch.float32
            ## 5. Feed the output and label to the loss_fn

            loss += loss_fn(out[is_labeled], batch.y[is_labeled].type(torch.float32))

            loss.backward()
            optimizer.step()

        return loss.item()