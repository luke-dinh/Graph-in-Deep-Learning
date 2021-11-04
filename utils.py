import numpy as np
import torch
import random
import torch.nn as nn

# Graph edge list and convert to tensor
def graph_to_edge_list(G):

    edge_list = list(G.edges())

    return edge_list

def edge_list_to_tensor(edge_list):

    edge_index = torch.tensor(np.array(edge_list), dtype=torch.long)
    edge_index = edge_index.T

    return edge_index 

# Negative Edges 
def sample_negative_edges(G, num_neg_samples):

    neg_edge_list = []

    pos_set = set(G.edges())
    visited_set = set()

    node_list = list(G.nodes())
    random.shuffle(node_list)

    for n_ith in node_list:
        for n_jth in node_list:

            # Check if the edges in pos_set or visited_set
            if n_ith == n_jth \
            or (n_ith, n_jth) in pos_set or (n_jth, n_ith) in pos_set \
            or (n_ith, n_jth) in visited_set or (n_jth, n_ith) in visited_set:
                continue

            neg_edge_list.append((n_ith, n_jth))
            visited_set.add((n_ith, n_jth))
            visited_set.add((n_jth, n_ith))

            if len(neg_edge_list) == num_neg_samples:
                return neg_edge_list

# Create node embedding matrix for the graph
torch.manual_seed(1)

def create_node_emb(num_node=34, embedding_dim=16):

    emb = nn.Embedding(num_embeddings=num_node, embedding_dim=embedding_dim)
    emb.weight.data = torch.randn(num_node, embedding_dim)

    return emb 