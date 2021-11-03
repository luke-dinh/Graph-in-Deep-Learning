import networkx as nx
import torch
import random
import numpy as np 

G = nx.karate_club_graph()

# Graph Edge list and list tensor
def graph_to_edge_list(G):

    edge_list = list(G.edges())

    return edge_list

def edge_list_to_tensor(edge_list):

    edge_index = torch.tensor(np.array(edge_list), dtype=torch.long)
    edge_index = edge_index.T

    return edge_index 

edge_list = graph_to_edge_list(G)
edge_index = edge_list_to_tensor(edge_list)

# Negative edges
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

# Sample 78 negative edges
neg_edge_list = sample_negative_edges(G, len(edge_list))

# Transform the negative edge list to tensor
neg_edge_index = edge_list_to_tensor(neg_edge_list)
print("The neg_edge_index tensor has shape {}".format(neg_edge_index.shape))

def is_neg_edge(edge):
    return not(edge in edge_list or (edge[1], edge[0]) in edge_list)

edge_1 = (7, 1)
edge_2 = (1, 33)
edge_3 = (33, 22)
edge_4 = (0, 4)
edge_5 = (4, 2)

print(is_neg_edge(edge_1))
print(is_neg_edge(edge_2))
print(is_neg_edge(edge_3))
print(is_neg_edge(edge_4))
print(is_neg_edge(edge_5))