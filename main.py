import matplotlib.pyplot as plt
import networkx as nx
import random
import itertools
import apx
import numpy as np
from pulp import *

g_n = 0
g_m = 0

plt.figure(figsize=(15,7))

############## Helper functions ##############

# Input
# G networkx graph object
# Output
# b_nodes list of boundary nodes
def get_boundary_nodes(G):
    b_nodes = list(filter(lambda t : t[0] in [0, g_m-1]  or t[1] in [0, g_n-1], G.nodes()))
    return b_nodes

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
# l: a list of lists
# Output
# Flat version of list l
def flatten_list(l):
    return [item for sublist in l for item in sublist]

############## Helper functions ##############

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
    nx.set_edge_attributes(G, 0, 'f_flow')
    nx.set_edge_attributes(G, 0, 'b_flow')
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
    node_labels=dict([(n,d['type']+':'+str(n)) for n,d in G.nodes(data=True)])
    edge_labels=dict([((u,v,),'('+str(d['f_flow'])+','+str(d['b_flow'])+') / '+str(d['capacity'])) for u,v,d in G.edges(data=True)])
    nx.draw(G,pos,font_size=8,node_color=node_cmap,edge_color=edge_cmap,node_size=1000,width=2,labels=node_labels,with_labels=True)
    nx.draw_networkx_edge_labels(G,pos,font_size=12,edge_color=edge_cmap,edge_labels=edge_labels)
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
            G.node[pair[0]]['type'] = 's'+str(i)
            G.node[pair[0]]['color'] = c_map[i]
            G.node[pair[1]]['type'] = 't'+str(i)
            G.node[pair[1]]['color'] = c_map[i]
    else:
        print('Error: k should be <= {0}'.format(len(b_nodes)))
    return (G, st_pairs)

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
# st_pairs List of source and sink pairs
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
                    G.edges[(path[i-1], path[i])]['f_flow'] += 1
                    G.edges[(path[i-1], path[i])]['color'] = G.node[path[0]]['color']
            break
    return G

# Input
# G networkx graph
# st_pairs List of source and sink pairs
# Output
# G after adding disjoint paths faster than exponential
def add_st_paths_fast(G, st_pairs):
    G_c = G.copy()

    for st_pair in st_pairs:
        try:
            path = list(nx.shortest_path(G_c, source=st_pair[0], target=st_pair[1]))
            print('Path chosen: ', path)

            for i in range(1, len(path)):
                G.edges[(path[i-1], path[i])]['f_flow'] += 1
                G.edges[(path[i-1], path[i])]['color'] = G.node[path[0]]['color']

            # Remove chosen path
            for e in to_edge_path(path):
                G_c.remove_edge(*e)
        except nx.exception.NetworkXNoPath:
            print('No path found for s-t pair:', st_pair)
    return G

# Input
# G networkx graph
# lp_dict dictionary of lp_paths
# st_pairs List of source and sink pairs
# Output
# Print paths to the screen
def print_paths(G, lp_dict, st_pairs):
    lp_dict = res = {k:v for k,v in lp_dict.items() if sum(list(map(lambda x : value(x), v))) != 0}

    for i, st_pair in enumerate(st_pairs):
        v = st_pair[0]
        path = [v]
        while v != st_pair[1]:
            for edge in G.edges(v):
                if edge in lp_dict.keys() and value(lp_dict[edge][i]) != 0:
                    path.append(edge[1])
                    v = edge[1]
        print('Path chosen: ', path)
    return


# Input
# G networkx graph
# st_pairs List of source and sink pairs
# Output
# LP values for the program
def run_LP(G, st_pairs, fn):
    prob = LpProblem("Minimize cost fn.", LpMinimize)
    k = len(st_pairs)               # num of s-t pairs

    # Creating 2k lp variables for every edge
    lp_dict = {}
    for i, edge in enumerate(G.edges()):
        temp1 = []
        temp2 = []
        for j in range(int(k)):
            temp1.append(LpVariable("edge"+str(i)+"flow_"+str(j), 0, None, LpInteger))
            temp2.append(LpVariable("egde"+str(i)+"flow_"+str(j), 0, None, LpInteger))
        lp_dict[edge] = temp1
        lp_dict[tuple(reversed(edge))] = temp2
    print('lp_dict', lp_dict)

    # List of lp_vars
    lp_vars = flatten_list(lp_dict.values())
    
    # Objecive fn
    prob += sum(list(map(lambda sl: fn['c'] + fn['m']*sum(sl), lp_dict.values())))

    print('adding constraints')
     # Add source/sink condition for their own flow
    print('st_pairs', st_pairs)
    for i, st_pair in enumerate(st_pairs):
        # sum of your own outgoing flows from source is 1
        temp1 = []
        temp2 = []
        for edge in G.edges(st_pair[0]):
            temp1.append(lp_dict[edge][i])
            temp2.append(lp_dict[tuple(reversed(edge))][i])
        prob += (sum(temp1) == 1)
        prob += (sum(temp2) == 0)

        # sum of your own incoming flows into sink is 1
        temp1 = []
        temp2 = []
        for edge in G.edges(st_pair[1]):
            temp1.append(lp_dict[edge][i])
            temp2.append(lp_dict[tuple(reversed(edge))][i])
        prob += (sum(temp1) == 0)
        prob += (sum(temp2) == 1)

    # Add source/sink conditions for other flows i.e those flow types should be preserved
    for i, st_pair in enumerate(st_pairs):
        # sum of other outgoing flows from source equals incoming flows
        temp1 = []
        temp2 = []
        for edge in G.edges(st_pair[0]):
            temp1.append(lp_dict[edge])                       # outgoing edges
            temp2.append(lp_dict[tuple(reversed(edge))])      # incoming edges
        for j in range(k):
            if j != i:
                temp1_k = list(map(lambda x: x[j], temp1))
                temp2_k = list(map(lambda x: x[j], temp2))
                prob += (sum(temp1_k) == sum(temp2_k))


        # sum of other outgoing flows from sink equals incoming flows
        temp1 = []
        temp2 = []
        for edge in G.edges(st_pair[1]):
            temp1.append(lp_dict[edge])                       # outgoing edges
            temp2.append(lp_dict[tuple(reversed(edge))])      # incoming edges
        for j in range(k):
            if j != i:
                temp1_k = list(map(lambda x: x[j], temp1))
                temp2_k = list(map(lambda x: x[j], temp2))
                prob += (sum(temp1_k) == sum(temp2_k))


    # Non-negativity constaint for every edge
    for var in lp_vars:
        prob += (var >= 0)

    # Zero-sum constraint for trauma edges
    for edge in G.edges(data = True):
        if edge[2]['capacity'] == 0:
            prob += (sum(lp_dict[(edge[0], edge[1])]) == 0)
            prob += (sum(lp_dict[(edge[1], edge[0])]) == 0)

    # For each internal node, flow conservation should hold for flow of each type
    temp1 = [] * k
    temp2 = [] * k
    for node in G.nodes(data = True):                            
        if 'n' in node[1]['type']:                                # node is not sink/source
            temp1 = []
            temp2 = []
            for edge in G.edges(node[0]):
                temp1.append(lp_dict[edge])                       # outgoing edges
                temp2.append(lp_dict[tuple(reversed(edge))])      # incoming edges
            for i in range(k):
                temp1_k = list(map(lambda x : x[i], temp1))
                temp2_k = list(map(lambda x : x[i], temp2))
                prob += (sum(temp1_k) == sum(temp2_k))
    print('adding constraints done')

    LpSolverDefault.msg = 1
    status = prob.solve()
    print("Status:", LpStatus[status])

    # Printing optimal values
    print('Optimal values')
    for k in lp_dict.keys():
        for v1 in lp_dict[k]:
            if value(v1) > 0:
                print(k, v1, value(v1))
    return lp_dict

# Input
# G networkx graph
# lp_dict dictionary of lp_paths
# st_pairs List of source and sink pairs
# Output
# G after updating flows according to the lp solution
def add_lp_paths(G, lp_dict, st_pairs):
    # Reset prev flows in graph to zeros
    for e in G.edges():
        G.edges[e]['f_flow'] = 0
        G.edges[e]['b_flow'] = 0
        G.edges[e]['color'] = (0, 0, 0)

    # Change flow values based on optimal lp_dict
    for edge in G.edges():
        ef_flows = lp_dict[edge]
        eb_flows = lp_dict[tuple(reversed(edge))]

        G.edges[edge]['f_flow'] =  G.edges[edge]['f_flow'] + sum(list(map(lambda x: value(x), ef_flows)))
        G.edges[edge]['b_flow'] =  G.edges[edge]['b_flow'] + sum(list(map(lambda x: value(x), eb_flows)))

    print_paths(G, lp_dict, st_pairs)
    return


# Input
# G networkx graph object
# Output
# Update graph and display it according to user input
def run(G):
    k_inp = None
    st_pairs = None
    lp_dict = None

    while True:
        prompt = "1: Set k value\n2: Delete a region\n3: Run LP\n"
        inp = input(prompt)

        if inp == "1":
            k_inp = eval(input('K : '))
            (G, st_pairs) = add_st_pairs(G, k_inp)
            G = add_st_paths_fast(G, st_pairs)
            display_graph(G)
        elif inp == "2":
            t_dim = eval(input('Size : '))
            G = add_trauma_grid(G, t_dim, t_dim)
            display_graph(G)
        elif inp == "3":
            lp_dict = run_LP(G, st_pairs)
            add_lp_paths(G, lp_dict, st_pairs)
            display_graph(G)
        else:
            print('Try again')
    return


def run_exps():
    # Exp 1
    G = setup_graph(5, 5)
    (G, st_pairs) = add_st_pairs(G, 8)
    G = add_st_paths_fast(G, st_pairs)
    G = add_trauma_grid(G, 3, 3)
    lp_dict = run_LP(G, st_pairs, {'c': 10, 'm': 5})
    add_lp_paths(G, lp_dict, st_pairs)
    # display_graph(G)

    # Exp 2
    G = setup_graph(5, 5)
    (G, st_pairs) = add_st_pairs(G, 8)
    G = add_st_paths_fast(G, st_pairs)
    G = add_trauma_grid(G, 3, 3)
    lp_dict = run_LP(G, st_pairs, {'c': 0, 'm': 5})
    add_lp_paths(G, lp_dict, st_pairs)
    # display_graph(G)

    # Exp 2
    G = setup_graph(5, 5)
    (G, st_pairs) = add_st_pairs(G, 8)
    G = add_st_paths_fast(G, st_pairs)
    G = add_trauma_grid(G, 3, 3)
    lp_dict = run_LP(G, st_pairs, apx.apx(lambda x: x**2 + 10, 2))
    add_lp_paths(G, lp_dict, st_pairs)
    # display_graph(G)


def main():
    # dim = eval(input('Grid Dimensions: '))
    # G = setup_graph(dim, dim)
    # display_graph(G, ion=True)
    # run(G)
    run_exps()


if __name__ == '__main__':
    main()