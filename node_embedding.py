import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from graph_to_tensor import *

G = nx.karate_club_graph()
# nx.draw(G, with_labels=True)

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

# Visualize node embeddings
def visualize_emb(emb):
  X = emb.weight.data.numpy()
  pca = PCA(n_components=2)
  components = pca.fit_transform(X)
  plt.figure(figsize=(6, 6))
  club1_x = []
  club1_y = []
  club2_x = []
  club2_y = []
  for node in G.nodes(data=True):
    if node[1]['club'] == 'Mr. Hi':
      club1_x.append(components[node[0]][0])
      club1_y.append(components[node[0]][1])
    else:
      club2_x.append(components[node[0]][0])
      club2_y.append(components[node[0]][1])
  plt.scatter(club1_x, club1_y, color="red", label="Mr. Hi")
  plt.scatter(club2_x, club2_y, color="blue", label="Officer")
  plt.legend()
  plt.show()

# Visualize the initial random embeddding
visualize_emb(emb)