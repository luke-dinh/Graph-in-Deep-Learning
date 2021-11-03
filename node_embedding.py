import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from graph_to_tensor import *

G = nx.karate_club_graph()

# # Example embedding long tensor
# emb_sample = nn.Embedding(num_embeddings=4, embedding_dim=8)
# ids = torch.LongTensor([1, 3])
# print(ids)
# print(emb_sample(ids))

# Create node embedding matrix for the graph
torch.manual_seed(1)

def create_node_emb(num_node=34, embedding_dim=16):

    emb = nn.Embedding(num_embeddings=num_node, embedding_dim=embedding_dim)
    emb.weight.data = torch.randn(num_node, embedding_dim)

    return emb 

emb = create_node_emb()
ids = torch.LongTensor([0, 3])

# Print the embedding layer
print("Embedding: {}".format(emb))

# Gets the embedding from node 0 and 3
print(emb(ids))