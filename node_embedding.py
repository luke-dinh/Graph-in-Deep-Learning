import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from graph_to_tensor import *

G = nx.karate_club_graph()

emb_sample = nn.Embedding(num_embeddings=4, embedding_dim=8)
