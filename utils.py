import numpy as np
import random

import torch 
import torch.nn as nn
from torch.optim import SGD

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

# Training the Embedding
class train_node_emb():

    def accuracy(self, pred, label):

        accu = 0.0

        pred = [1 if item>0.5 else 0 for item in pred]
        matching = (np.array(pred) == np.array(label)).sum()
        accu += round(matching/len(label), 4)

        return accu 

    def train(self, emb, loss_fn, sigmoid, train_label, train_edge):

        epochs = 500
        learning_rate = 0.1

        optimizer = SGD(emb.parameters(), lr=learning_rate, momentum=0.9)

        for i in range(epochs):

            # Get the embeddings of the nodes in train_edge
            emb_u = emb(train_edge)[0]
            emb_v = emb(train_edge)[1]

            # Sum of dot product the embeddings of the nodes in train_edge
            dot_product = torch.sum(emb_u * emb_v, dim=-1)

            # Feed the dot product into sigmoid function
            sig = sigmoid(dot_product)

            # Feed the sigmoid output into loss_fn
            loss = loss_fn(sig, train_label)

            # Update the embeddings

            loss.backward() # Backprop
            optimizer.step() # Update parameter

            # Print both loss and acc of each epoch

            print("Loss for epoch {} : ".format(i), loss)
            print("Accuracy for epoch {} is: ".format(i), self.accuracy(sig, train_label))