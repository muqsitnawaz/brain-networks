import matplotlib.pyplot as plt
import networkx as nx
import random
import itertools

g_n = 0
g_m = 0

plt.figure(figsize=(15,7))

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
    nx.set_edge_attributes(G, (0, 0, 0), 'color')
    
    return G

# Input
# G networkx graph object
# Output
# Displays the graph in matplotlib window
def display_graph(G, ion=False,relabel=False):
    plt.gcf().clear()
    
    pos = {}
    for node in G.nodes():
        pos[node] = (node[1]*5, (g_m-1)*5-node[0]*5)
    
    node_cmap = list(map(lambda x: x[1]['color'],G.nodes(data=True)))
    edge_cmap = list(map(lambda e: e[2]['color'],G.edges(data=True)))


    if relabel == True:
        G = nx.convert_node_labels_to_integers(G, ordering = 'sorted')
    edge_labels=dict([((u,v,),str(d['flow'])+'/'+str(d['capacity'])) for u,v,d in G.edges(data=True)])

    nx.draw(G,pos,font_size=8,node_color=node_cmap,edge_color=edge_cmap,node_size=500,width=2,with_labels=True)
    nx.draw_networkx_edge_labels(G,pos,font_size=8,edge_color=edge_cmap,edge_labels=edge_labels)
    if ion == True:
        plt.ion()
    plt.show()

# Input
# G networkx graph object
# t_m, t_m dimension of rectangular trauma grid
# Output
# G after adding trauma grid and set cap of edges inside T to 0
def add_trauma_grid(G, t_m, t_n):
    rand_row = random.randint(1, g_m-t_m-1)
    rand_col = random.randint(1, g_m-t_n-1)

    trauma_nodes = list(itertools.product(range(rand_row, rand_row+t_m), range(rand_col, rand_col+t_n)))
    for node in trauma_nodes:
        G.node[node]['color'] = (1, 0, 0)

    trauma_edges = list(filter(lambda e : e[0] in trauma_nodes and e[1] in trauma_nodes, G.edges()))
    for edge in trauma_edges:
        G.edges[edge]['capacity'] = 0
    return G

# Input
# G networkx graph object
# Output
# b_nodes list of boundary nodes
def get_boundary_nodes(G):
    b_nodes = list(filter(lambda t : t[0] in [0, g_m-1]  or t[1] in [0, g_n-1], G.nodes()))
    return b_nodes

# Input
# G networkx graph object
# k number of source/sink pairs
# Output
# G after adding s-t pairs and color coding them
def add_st_pairs(G, k):
    b_nodes = get_boundary_nodes(G)
    st_pairs = None

    if 2*k <= len(b_nodes):
        r_nodes = random.sample(b_nodes, 2*k)

        random.shuffle(r_nodes)
        st_pairs = list(zip(r_nodes[:k], r_nodes[k:]))
        c_map = [(random.random(), 0.5*random.random(), random.random()) for i in range(len(st_pairs))]
        
        for (i, pair) in enumerate(st_pairs):
            G.node[pair[0]]['type'] = 's'
            G.node[pair[0]]['color'] = c_map[i]
            G.node[pair[1]]['type'] = 't'
            G.node[pair[1]]['color'] = c_map[i]
    else:
        print('Error: k should be <= {0}'.format(len(b_nodes)))
    return (G, st_pairs)

# Input
# p A list of nodes which represent a path from start to end in networkx graph
# Output
# path of edges made from the path of nodes
def to_edge_path(p):
    path = []
    for i in range(1,len(p)):
        path.append((p[i-1],p[i]))
    return path

# Input
# paths A list of node-paths to check for disjoint-ness i.e no edge occurs twice
# edges The list of the edges in the graph
# Output
# True If list of paths is disjoint otherwise false
def is_disjoint_comb(paths, edges):
    paths = list(map(lambda p: to_edge_path(p), paths))
    
    used = {}
    for edge in edges:
        used[edge] = False
        used[tuple(reversed(edge))] = False

    for path in paths:
        for edge in path:
            if used[edge] or used[tuple(reversed(edge))]:
                return False
            else:
                used[edge] = True
                used[tuple(reversed(edge))] = True
    return True


# Input
# G networkx graph
# st_pairs List of st_pairs
# Output
# G after adding disjoint paths if possible
def add_st_paths(G, st_pairs):
    st_paths = list(map(lambda p: list(nx.all_simple_paths(G, source=p[0], target=p[1])), st_pairs))
    all_combs = list(itertools.product(*st_paths))

    for c in all_combs:
        if is_disjoint_comb(c, G.edges()):
            for path in c:
                print('Path chosen: ', path)
                for i in range(1, len(path)):
                    G.edges[(path[i-1], path[i])]['flow'] += 1
                    G.edges[(path[i-1], path[i])]['color'] = G.node[path[0]]['color']
            break
    return G

# Input
# G networkx graph object
# Output
# Update graph according to user input
def run(G):
    while True:
        prompt = "1: Set k value\n2: Delete a region\n"
        inp = input(prompt)

        if inp == "1":
            k_inp = eval(input('K : '))
            (G, st_pairs) = add_st_pairs(G, k_inp)
            G = add_st_paths(G, st_pairs)
            display_graph(G)
        elif inp == "2":
            t_dim = eval(input('Size : '))
            G = add_trauma_grid(G, t_dim, t_dim)
            display_graph(G)
        else:
            print('Try again')


def main():
    dim = eval(input('Grid Dimensions: '))
    # dim = 5
    G = setup_graph(dim, dim)
    display_graph(G, ion=True)
    run(G)

if __name__ == '__main__':
    main()