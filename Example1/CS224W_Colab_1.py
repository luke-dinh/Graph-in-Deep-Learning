import networkx as nx

# Import Graph
G = nx.karate_club_graph()
nx.draw(G, with_labels=True)

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

# Demo PageRank Equation
def one_iter_pagerank(G, beta, r0, node_id):

    r1 = 0
    N = G.number_of_nodes()
    for node_neighbor in G.neighbors(node_id):
        node_deg = G.degree[node_neighbor]
        r1 += beta * r0/node_deg + (1 - beta)/N 
    
    return round(r1,2) 

# Raw closeness centrality
def closeness_centrality(G, node=5):

    closeness = 0
    total_shortest = 0
    for path in list(nx.single_source_shortest_path(G, node).values())[1:]:
        total_shortest += len(path) - 1

    closeness += 1 /total_shortest
    return round(closeness, 2)

beta = 0.8
r0 = 1 / G.number_of_nodes()
node = 0

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
average_degree = avg_degree(num_nodes, num_edges)
avg_cluster_coef = avg_coefficient(G)
iter_pagerank = one_iter_pagerank(G, beta, r0, node)
avg_shortest = closeness_centrality(G, node=5)

print(average_degree)
print(avg_cluster_coef)
print(iter_pagerank)
print(avg_shortest)