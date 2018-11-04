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
    if relabel == True:
        G = nx.convert_node_labels_to_integers(G, ordering = 'sorted')
    
    pos = {}
    for node in G.nodes():
        pos[node] = (node[1]*5, (g_m-1)*5-node[0]*5)
    
    node_cmap = list(map(lambda x: x[1]['color'],G.nodes(data=True)))
    edge_cmap = list(map(lambda e: e[2]['color'],G.edges(data=True)))
    nx.draw(G,pos,font_size=8,node_color=node_cmap,edge_color=edge_cmap,with_labels=True)

    edge_labels=dict([((u,v,),str(d['flow'])+'/'+str(d['capacity'])) for u,v,d in G.edges(data=True)])
    nx.draw_networkx_edge_labels(G,pos,font_size=8,edge_color=edge_cmap,edge_labels=edge_labels)
    

    plt.rcParams["figure.figsize"] = (20, 20)
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
    # nodes = [x for x in range()]
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

def convert_node_path_to_edge_path(p):
    path = []
    for i in range(1,len(p)):
        path.append((p[i-1],p[i]))
    return path

def is_disjoint_paths(path1, path2):
    return


def find_disjoint_paths(paths, edges):
    paths = list(map(lambda p: convert_node_path_to_edge_path(p), paths))

    used = {}
    for edge in edges:
        used[edge] = False
        used[(edge[1], edge[0])] = False

    d_paths = []
    for i in range(len(paths)):
        for j in range(i, len(paths)):
            if [x for x in paths[i] if x in paths[j]] == []:
                if not (any([used[x] for x in paths[i]]) or any([used[x] for x in paths[j]])):
                    for e in paths[i]:
                        used[e] = True
                    for e in paths[j]:
                        used[e] = True
                    d_paths.extend([paths[i], paths[j]])
    return d_paths


def choose_disjoint_paths(st_pairs, paths):
    c_paths = {}
    for p in st_pairs:
        c_paths[p] = []

    for p in paths:
        c_paths[(p[0][0],p[-1][1])] += [p]

    # for st_pair in st_pairs:
    #     for path in paths:
    #         if st_pair[0] == path[0] and st_pair[1] == path[-1]:
    #             c_paths.append(path)
    #             break
    return c_paths



def add_st_paths(G, st_pairs):
    st_paths = list(map(lambda p: list(nx.all_simple_paths(G, source=p[0], target=p[1])), st_pairs))
    # temp = [path for paths in st_paths for path in paths]
    # for st_path in st_paths:
    #     temp.extend(st_path)
    d_paths = find_disjoint_paths(temp, G.edges())
    print(d_paths)
    c_paths = choose_disjoint_paths(st_pairs,d_paths)
    print(c_paths)
    # if len(c_paths) == len(st_pairs):
    #     for path in c_paths:
    #         for i in range(1, len(path)):
    #             G.edges[(path[i-1], path[i])]['flow'] = 1
    # else:
    #     print('Error: No disjoint s-t paths found')
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
            (G, st_pairs) = add_st_pairs(G, k_inp)
            G = add_st_paths(G, st_pairs)
            display_graph(G)
        elif inp == "2":
            t_dim = eval(input('Size : '))
            G = add_trauma_grid(G, t_dim, t_dim)
            display_graph(G)


def main():
    dim = eval(input('Grid Dimensions: '))
    # dim = 5
    G = setup_graph(dim, dim)
    display_graph(G, ion=True)
    run(G)

if __name__ == '__main__':
    main()