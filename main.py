import matplotlib.pyplot as plt
import networkx as nx
import random
import itertools

g_n = 0
g_m = 0

# Input
# n, m where n and m are dimension of the grid
# Output
# Returns a networkx graph object
def setup_graph(n, m):
    global g_n, g_m
    g_n = n
    g_m = m
    G = nx.grid_2d_graph(n, m)

    nx.set_node_attributes(G, (0, 1, 0), 'color')
    nx.set_node_attributes(G, 'n', 'type')

    nx.set_edge_attributes(G, 1, 'capacity')
    nx.set_edge_attributes(G, 0, 'flow')
    
    return G

# Input
# G networkx graph object
# Output
# Displays the graph in matplotlib window
def display_graph(G, ion=False,relabel=False):
    if relabel == True:
        G = nx.convert_node_labels_to_integers(G, ordering = 'sorted')
    
    pos = {}
    for node in G.nodes():
        pos[node] = (node[1]*5, (g_m-1)*5-node[0]*5)
    
    color_map = list(map(lambda x: x[1]['color'],G.nodes(data=True)))
    nx.draw(G,pos,node_color=color_map,font_size=8,with_labels=True)

    edge_labels=dict([((u,v,),str(d['flow'])+'/'+str(d['capacity'])) for u,v,d in G.edges(data=True)])
    nx.draw_networkx_edge_labels(G,pos,font_size=8,edge_labels=edge_labels)
    

    if ion == True:
        plt.ion()
    plt.show()

# Input
# G networkx graph object
# Output
# Display the graph in current matplotlib window
def update_graph(G):
    plt.gcf().clear()
    display_graph(G)

# Input
# G networkx graph object
# t_m, t_m dimension of rectangular trauma grid
# Output
# G after adding trauma grid and set cap of edges inside T to 0
def add_trauma_grid(G, t_m, t_n):
    # print(G.nodes(data=True))
    rand_row = random.randint(1, g_m-t_m-1)
    rand_col = random.randint(1, g_m-t_n-1)
    print(rand_row,rand_col)

    trauma_nodes = list(itertools.product(range(rand_row, rand_row+t_m), range(rand_col, rand_col+t_n)))
    for node in trauma_nodes:
        G.node[node]['color'] = (1, 0, 0)

    trauma_edges = list(filter(lambda e : e[0] in trauma_nodes and e[1] in trauma_nodes, G.edges()))
    for edge in trauma_edges:
        G.edges[edge]['capacity'] = 0
    return G

def get_boundary_nodes(G):
    b_nodes = list(filter(lambda t : t[0] in [0, g_m-1]  or t[1] in [0, g_n-1], G.nodes()))
    return b_nodes

# Input
# G networkx graph object
# k number of source/sink pairs
# Output
# G after adding trauma grid and set cap of edges inside T to 0
def add_st_pairs(G, k):
    # nodes = [x for x in range()]
    b_nodes = get_boundary_nodes(G)
    if 2*k <= len(b_nodes):
        r_nodes = random.sample(b_nodes, 2*k)

        random.shuffle(r_nodes)
        pairs = list(zip(r_nodes[:k], r_nodes[k:]))
        c_map = [(random.random(), 0.5*random.random(), random.random()) for i in range(len(pairs))]
        
        for (i, pair) in enumerate(pairs):
            G.node[pair[0]]['type'] = 's'
            G.node[pair[0]]['color'] = c_map[i]
            G.node[pair[1]]['type'] = 't'
            G.node[pair[1]]['color'] = c_map[i]
    else:
        print('Error: k should be <= {0}'.format(len(b_nodes)))
    return G

# Input
# G networkx graph object
# Output
# Update graph according to user input
def run(G):
    while True:
        prompt = "1: Set k value\n2: Delete a region\n3:"
        inp = input(prompt)

        if inp == "1":
            k_inp = eval(input('K : '))
            G = add_st_pairs(G, k_inp)
            update_graph(G)
        elif inp == "2":
            t_dim = eval(input('Size : '))
            G = add_trauma_grid(G, t_dim, t_dim)
            update_graph(G)


def main():
    # dim = eval(input('dimensions? '))
    dim = 5
    G = setup_graph(dim, dim)
    display_graph(G, ion=True)
    run(G)

if __name__ == '__main__':
    main()