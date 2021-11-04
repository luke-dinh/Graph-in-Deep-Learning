import networkx as nx
import matplotlib.pyplot as plt

# Define undirected graph
G = nx.Graph()

# Add nodes
G.add_node(1)
G.add_nodes_from([2,3,4])
G.add_nodes_from([ 
    (4, {"name": "Luke"}),
    (5, {"name": "Tom"})
])

# Add edges
G.add_edge(1, 3)
G.add_edges_from([(1,5), (2,3)])
G.add_edge(2,5)
G.add_edge(4,5)

# Visualize the graph
plt.figure(figsize=(10,10))
nx.draw(G, with_labels=True)
plt.show()