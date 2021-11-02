import networkx as nx

# Import Graph
G = nx.karate_club_graph()

# Average degree
def avg_degree(num_edges, num_nodes):

    avg_degree = 0
    avg_degree += round((2 * num_edges / num_nodes),2)

    return avg_degree 

# Average clustering coefficient
def avg_coefficient(G):

    avg_cluster_coef = 0
    avg_cluster_coef += round(nx.average_clustering(G), 2)

    return avg_cluster_coef

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
average_degree = avg_degree(num_nodes, num_edges)
avg_cluster_coef = avg_coefficient(G)
print(average_degree)
print(avg_cluster_coef)