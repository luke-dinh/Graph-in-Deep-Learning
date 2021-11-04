import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.nn.modules.activation import Sigmoid
from torch.optim import SGD

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

# # Visualize node embeddings
# def visualize_emb(emb):
#   X = emb.weight.data.numpy()
#   pca = PCA(n_components=2)
#   components = pca.fit_transform(X)
#   plt.figure(figsize=(6, 6))
#   club1_x = []
#   club1_y = []
#   club2_x = []
#   club2_y = []
#   for node in G.nodes(data=True):
#     if node[1]['club'] == 'Mr. Hi':
#       club1_x.append(components[node[0]][0])
#       club1_y.append(components[node[0]][1])
#     else:
#       club2_x.append(components[node[0]][0])
#       club2_y.append(components[node[0]][1])
#   plt.scatter(club1_x, club1_y, color="red", label="Mr. Hi")
#   plt.scatter(club2_x, club2_y, color="blue", label="Officer")
#   plt.legend()
#   plt.show()

# # Visualize the initial random embeddding
# visualize_emb(emb)

# Training the Embedding

def accuracy(pred, label):

  accu = 0.0

  pred = [1 if item>0.5 else 0 for item in pred]
  matching = (np.array(pred) == np.array(label)).sum()
  accu += round(matching/len(label), 4)

  return accu 

def train(emb, loss_fn, sigmoid, train_label, train_edge):

  epochs = 100
  learning_rate = 0.1

  optimizer = SGD(emb.parameters(), lr=learning_rate, momentum=0.9)

  for i in range(epochs):

    # Get the embeddings of the nodes in train_edge
    emb_u = emb(train_edge)[0]
    emb_v = emb(train_edge)[1]

    # Sum of dot product the embeddings of the nodes in train_edge
    dot_product = torch.sum(np.dot(emb_u, emb_v), dim=-1)

    # Feed the dot product into sigmoid function
    sig = sigmoid(dot_product)

    # Feed the sigmoid output into loss_fn
    loss = loss_fn(sig, train_label)

    # Print both loss and acc of each epoch

    print("Loss for epoch {i} : {}".format(loss))
    print("Accuracy for epoch {i} is: {}".format(accuracy(sig, train_label)))

    # Update the embeddings

    loss.backward() # Backprop
    optimizer.step() # Update parameter

loss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()

pos_edge_index = edge_list_to_tensor(graph_to_edge_list(G))
neg_edge_index = edge_list_to_tensor(sample_negative_edges(G, len(graph_to_edge_list(G))))

