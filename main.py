import matplotlib.pyplot as plt
import networkx as nx
import random
import itertools
import apx
import time
import math
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


# Input
# G: networkx graph object
# Output
# Reset flows in the graph
def reset_flows(G):
    for edge in G.edges():
        G.edges[edge]['f_flow'] = 0
        G.edges[edge]['b_flow'] = 0
    return G

############## Helper functions ##############

############## File writing functions ##############
# Input
# dim_x dimension of Graph G
# dim_y dimension of Graph G
# cost_fn 
# f output file object
# Output
# Write statistics about graph to input file f
def write_pre_trauma(G, st_pairs, cost_fn, f):
    g_n = math.sqrt(len(G.nodes()))
    f.write('Grid dimensions: ' + str(g_n) + ' ' + str(g_n) + '\n')
    f.write('Total number of nodes: {}\n'.format(g_n**2))
    f.write('Total number of edges: {}\n'.format(2*g_n*(g_n - 1)))
    f.write('Cost fn: y = {}x + {}\n'.format(cost_fn['m'], cost_fn['c']))
    f.write('s-t pairs: {}\n'.format(str(st_pairs)))
    return

# Input
# G networkx graph object
# cost_fn cost function
# f output file
# Output
# Write statistics about graph to input file f
def write_post_trauma(G, cost_fn, f):
    num_edges = 0
    num_trauma_edges = 0
    tot_cost = 0
    
    for edge in G.edges(data = True):
        tot_flow = edge[2]['f_flow'] + edge[2]['b_flow']

        if edge[2]['capacity'] == 0:
            num_trauma_edges += 1

        if tot_flow > 1:
            G.edges[(edge[0], edge[1])]['capacity'] = tot_flow
            tot_cost += cost_fn['c'] + cost_fn['m']*tot_flow
            num_edges += 1

    f.write('Number of trauma edges: {}\n'.format(num_trauma_edges))
    f.write('Number of edges whose capacity was increased: {}\n'.format(num_edges))
    f.write('Cost of LP solution: ' + str(tot_cost) + '\n')
    return


# Input
# G networkx graph
# lp_dict dictionary of lp_paths
# st_pairs List of source and sink pairs
# Output
# Print paths to the screen
def write_lp_paths(G, lp_dict, st_pairs, f=None):
    lp_dict = {k:v for k,v in lp_dict.items() if sum(list(map(lambda x : value(x), v))) != 0}

    st_paths = {}
    for i, st_pair in enumerate(st_pairs):
        v = st_pair[0]
        path = [v]
        while v != st_pair[1]:
            for edge in G.edges(v):
                if edge in lp_dict.keys() and value(lp_dict[edge][i]) != 0:
                    path.append(edge[1])
                    v = edge[1]
        st_paths[st_pair] = path
        print('Path', st_pair, ' => ', path)

        # Write path to file
        if f is not None:
            f.write('Path ' + str(st_pair) + ' => ' + str(path) + '\n')
    return st_paths

############## File writing functions ##############

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
    return


# Input
# G networkx graph object
# Output
# Saves the graph to a file named figname
def save_graph(G, figpath, ion=False, relabel=False):
    plt.gcf().clear()
    fig, ax = plt.subplots(nrows=1, ncols=1)

    # Add labels and colors to graph
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

    # Set fig size and save fig
    fig.set_size_inches(math.sqrt(len(G.nodes()))*2.5, math.sqrt(len(G.nodes()))*2.5)
    fig.savefig(figpath)
    return

# Input
# G networkx graph object
# Output
# Saves the graph to a file named figname
def save_pretty_graph(G, st_paths, lp_dict, figpath, ion=False, relabel=False):
    for (st_pair, st_path) in st_paths.items():
        # Set color of edges according to st_pairs
        st_path = to_edge_path(st_path)

        for edge in st_path:
            ef_flows = lp_dict[edge]
            eb_flows = lp_dict[tuple(reversed(edge))]

            G.edges[edge]['color'] = G.node[st_pair[0]]['color']
            G.edges[edge]['f_flow'] = G.edges[edge]['f_flow'] + sum(list(map(lambda x: value(x), ef_flows)))
            G.edges[edge]['b_flow'] =  G.edges[edge]['b_flow'] + sum(list(map(lambda x: value(x), eb_flows)))

        # Now plot the graph and save
        plt.gcf().clear()
        fig, ax = plt.subplots(nrows=1, ncols=1)

        # Add labels and colors to graph
        pos = {}
        for node in G.nodes():
            pos[node] = (node[1]*5, (g_m-1)*5-node[0]*5)
        node_cmap = list(map(lambda x: x[1]['color'],G.nodes(data=True)))
        edge_cmap = list(map(lambda e: e[2]['color'],G.edges(data=True)))
        node_labels=dict([(n,d['type']+':'+str(n)) for n,d in G.nodes(data=True)])
        edge_labels=dict([((u,v,),'('+str(d['f_flow'])+','+str(d['b_flow'])+') / '+str(d['capacity'])) for u,v,d in G.edges(data=True)])
        nx.draw(G,pos,font_size=8,node_color=node_cmap,edge_color=edge_cmap,node_size=1000,width=5,labels=node_labels,with_labels=True)
        nx.draw_networkx_edge_labels(G,pos,font_size=12,edge_color=edge_cmap,edge_labels=edge_labels)
        if ion == True:
            plt.ion()

        # Set fig size and save fig
        fig.set_size_inches(math.sqrt(len(G.nodes()))*2.5, math.sqrt(len(G.nodes()))*2.5)
        fig.savefig(figpath+'-path-{}.png'.format(G.node[st_pair[0]]['type']))

        for edge in G.edges():
            G.edges[edge]['color'] = (0, 0, 0)
    return

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
# LP values for the program
def check_feasible(G, st_pairs):
    prob = LpProblem("Check feasible", LpMinimize)
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

    # List of lp_vars
    lp_vars = flatten_list(lp_dict.values())
    
    # Objecive fn
    prob += sum(lp_vars)

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

    # For each edge, sum of all flow types should be at most 1
    for edge in G.edges():
        temp1 = lp_dict[edge]
        temp2 = lp_dict[tuple(reversed(edge))]
        prob += ((sum(temp1) + sum(temp2)) <= 1)
    print('adding constraints done')

    LpSolverDefault.msg = 1
    prob.writeLP("feasible.txt")
    status = prob.solve()
    print("Status:", LpStatus[status])

    # Printing optimal values
    print('Optimal values')
    for k in lp_dict.keys():
        for v1 in lp_dict[k]:
            if value(v1) > 0:
                print(k, v1, value(v1))
    return (LpStatus[status], lp_dict)

# Input
# G networkx graph
# st_pairs List of source and sink pairs
# Output
# LP values for the program
def run_LP_linear(G, st_pairs, fn):
    prob = LpProblem("Minimize cost fn.", LpMinimize)
    k = len(st_pairs)

    # Create 2k lp variables for flows on every edge
    lp_dict = {}
    for i, edge in enumerate(G.edges()):
        temp1 = []
        temp2 = []
        for j in range(int(k)):
            temp1.append(LpVariable("edge"+str(i)+"flow_"+str(j), 0, None, LpInteger))
            temp2.append(LpVariable("egde"+str(i)+"flow_"+str(j), 0, None, LpInteger))
        lp_dict[(edge[0], edge[1])] = temp1
        lp_dict[(edge[1], edge[0])] = temp2

    # Create 2k lp variables for slack on every edge
    lp_dict_s = {}
    for i, edge in enumerate(G.edges()):
        temp1 = []
        temp2 = []
        for j in range(int(k)):
            temp1.append(LpVariable("slk_edge"+str(i)+"flow_"+str(j), 0, None, LpInteger))
            temp2.append(LpVariable("slk_egde"+str(i)+"flow_"+str(j), 0, None, LpInteger))
        lp_dict_s[(edge[0], edge[1])] = temp1
        lp_dict_s[(edge[1], edge[0])] = temp2
    
    # Objecive fn
    prob += sum(list(map(lambda sl: fn['m']*sum(sl), lp_dict_s.values())))

    print('adding constraints')
    # Add slack constraints
    for edge in G.edges():
        r_edge = tuple(reversed(edge))
        temp1 = []
        temp2 = []
        temp3 = []
        temp4 = []
        for i in range(k):
            temp1.append(lp_dict[edge][i])
            temp1.append(lp_dict[r_edge][i])

            temp2.append(lp_dict_s[edge][i])
            temp2.append(lp_dict_s[r_edge][i])
        prob += (sum(temp1) - sum(temp2) <= 1)

     # Add source/sink condition for their own flow
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


    # Non-neg constraint for every edge
    for var in flatten_list(lp_dict.values()):
        prob += (var >= 0)

    # Non-neg constraint for every slack variable
    for var in flatten_list(lp_dict_s.values()):
        prob += (var >= 0)

    # Zero-sum constraint for trauma edges
    for edge in G.edges(data = True):
        if edge[2]['capacity'] == 0:
            prob += (sum(lp_dict[(edge[0], edge[1])]) == 0)
            prob += (sum(lp_dict[(edge[1], edge[0])]) == 0)
            prob += (sum(lp_dict_s[(edge[0], edge[1])]) == 0)
            prob += (sum(lp_dict_s[(edge[1], edge[0])]) == 0)

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
    prob.writeLP("trauma.txt")
    status = prob.solve()

    print("Status:", LpStatus[status])

    # Printing optimal values
    print('Optimal flow values')
    for k in lp_dict.keys():
        for v1 in lp_dict[k]:
            if value(v1) > 0:
                print(k, v1, value(v1))
    print('Optimal slack values')
    for k in lp_dict_s.keys():
        for v1 in lp_dict_s[k]:
            if value(v1) > 0:
                print(k, v1, value(v1))
    return (lp_dict, lp_dict_s)

# Input
# G networkx graph
# lp_dict dictionary of lp_paths
# st_pairs List of source and sink pairs
# Output
# G after updating flows according to the lp solution
def add_lp_paths(G, lp_dict, st_pairs):
    # Change flow values based on optimal lp_dict
    for edge in G.edges():
        ef_flows = lp_dict[edge]
        eb_flows = lp_dict[tuple(reversed(edge))]

        G.edges[edge]['f_flow'] =  G.edges[edge]['f_flow'] + sum(list(map(lambda x: value(x), ef_flows)))
        G.edges[edge]['b_flow'] =  G.edges[edge]['b_flow'] + sum(list(map(lambda x: value(x), eb_flows)))
    return

# Input
# None
# Output
# Run LP solver to get optimal flow values
def run_trauma_exps():
    for g in range(4, 6):
        for k in range(1, 2*g-1):
            for t in range(2, g-1):
                print('g', g, 'k', k, 't', t)
                for i in range(0, 50):
                    f = open("./results/g-{}-k-{}-t-{}-iter-{}.exp".format(g, k, t, i), "w")
                    cost_fn = {'c': 10, 'm': 5}

                    # Setup graph
                    G = setup_graph(g, g)
                    (G, st_pairs) = add_st_pairs(G, k)

                    # Add initial paths
                    write_pre_trauma(G, st_pairs, cost_fn, f)
                    (status, lp_dict_f) = check_feasible(G, st_pairs)
                    add_lp_paths(G, lp_dict_f, st_pairs)
                    f.write('Initial problem status (using LP): {}\n'.format(status))
                    # save_graph(G, "./exp1-bf-trauma")

                    # Add trauma to graph
                    G = add_trauma_grid(G, t, t)
                    st_time = time.time()
                    (lp_dict_f, lp_dict_s) = run_LP_linear(G, st_pairs, {'c': 10, 'm': 5})
                    lp_time = time.time() - st_time
                    f.write('Runtime of LP: {}\n'.format(lp_time))

                    # Write post trauma graph statistics
                    G = reset_flows(G)
                    add_lp_paths(G, lp_dict_f, st_pairs)
                    write_post_trauma(G, cost_fn, f)
                    # save_graph(G, "./exp1-af-trauma")
                    f.close()
    return

def run_runtime_exps():
    for g in range(20, 21):
        k = g
        t = '3-tr'
        print('g', g, 'k', k, 't', t)
        for i in range(0, 30):
            f = open("./results/runtime/g-{}-k-{}-t-{}-iter-{}.exp".format(g, k, t, i), "w")
            cost_fn = {'c': 10, 'm': 5}

            # Setup graph
            G = setup_graph(g, g)
            (G, st_pairs) = add_st_pairs(G, k)

            # Add initial paths
            write_pre_trauma(G, st_pairs, cost_fn, f)
            (status, lp_dict_f) = check_feasible(G, st_pairs)
            add_lp_paths(G, lp_dict_f, st_pairs)
            f.write('Initial problem status (using LP): {}\n'.format(status))
            save_graph(G, "./results/runtime/g-{}-k-{}-t-{}-iter-{}-bf.png".format(g, k, t, i))

            # Add trauma to graph
            G = add_trauma_grid(G, 2, 2)
            G = add_trauma_grid(G, 4, 4)
            G = add_trauma_grid(G, 3, 3)
            st_time = time.time()
            (lp_dict_f, lp_dict_s) = run_LP_linear(G, st_pairs, {'c': 10, 'm': 5})
            lp_time = time.time() - st_time
            f.write('Runtime of LP: {}\n'.format(lp_time))

            # Write post trauma graph statistics
            G = reset_flows(G)
            add_lp_paths(G, lp_dict_f, st_pairs)
            write_post_trauma(G, cost_fn, f)
            save_graph(G, "./results/runtime/g-{}-k-{}-t-{}-iter-{}-af.png".format(g, k, t, i))
            f.close()
    return

def run_pretty_paths_exps():
    g = 8
    k = g
    t = 'm'
    print('g', g, 'k', k, 't', t)
    for i in range(0, 1):
        f = open("./results/paths/g-{}-k-{}-t-{}-iter-{}.exp".format(g, k, t, i), "w")
        cost_fn = {'c': 10, 'm': 5}

        # Setup graph
        G = setup_graph(g, g)
        (G, st_pairs) = add_st_pairs(G, k)

        # Add initial paths
        write_pre_trauma(G, st_pairs, cost_fn, f)
        (status, lp_dict_f) = check_feasible(G, st_pairs)
        f.write('Initial problem status (using LP): {}\n'.format(status))
        st_paths = write_lp_paths(G, lp_dict_f, st_pairs, f)
        save_pretty_graph(G, st_paths, lp_dict_f, "./results/paths/g-{}-k-{}-t-{}-iter-{}-bf".format(g, k, t, i))

        # Add trauma to graph
        G = add_trauma_grid(G, 2, 2)
        G = add_trauma_grid(G, 4, 4)
        G = add_trauma_grid(G, 3, 3)
        st_time = time.time()
        (lp_dict_f, lp_dict_s) = run_LP_linear(G, st_pairs, {'c': 10, 'm': 5})
        lp_time = time.time() - st_time
        f.write('paths of LP: {}\n'.format(lp_time))

        # Write post trauma graph statistics
        G = reset_flows(G)
        write_post_trauma(G, cost_fn, f)
        st_paths = write_lp_paths(G, lp_dict_f, st_pairs, f)
        save_pretty_graph(G, st_paths, lp_dict_f, "./results/paths/g-{}-k-{}-t-{}-iter-{}-af".format(g, k, t, i))
        f.close()
    return

def run_st_pairs_exps():
    g = 10
    k, t = g, 'm'

    print('g', g, 'k', k, 't', t)
    for k in range(1, 2*g-1):
        for i in range(0, 5):
            f = open("./results/st-pairs/g-{}-k-{}-t-{}-iter-{}.exp".format(g, k, t, i), "w")
            cost_fn = {'c': 10, 'm': 5}

            # Setup graph
            G = setup_graph(g, g)
            (G, st_pairs) = add_st_pairs(G, k)

            # Add initial paths
            write_pre_trauma(G, st_pairs, cost_fn, f)
            (status, lp_dict_f) = check_feasible(G, st_pairs)
            add_lp_paths(G, lp_dict_f, st_pairs)
            f.write('Initial problem status (using LP): {}\n'.format(status))
            save_graph(G, "./results/st-pairs/g-{}-k-{}-t-{}-iter-{}-bf.png".format(g, k, t, i))

            # Add trauma to graph
            for i in range(random.randint(2, 5)):
                G = add_trauma_grid(G, 3, 3)

            st_time = time.time()
            (lp_dict_f, lp_dict_s) = run_LP_linear(G, st_pairs, {'c': 10, 'm': 5})
            lp_time = time.time() - st_time
            f.write('Runtime of LP: {}\n'.format(lp_time))

            # Write post trauma graph statistics
            G = reset_flows(G)
            add_lp_paths(G, lp_dict_f, st_pairs)
            write_post_trauma(G, cost_fn, f)
            save_graph(G, "./results/st-pairs/g-{}-k-{}-t-{}-iter-{}-af.png".format(g, k, t, i))
            f.close()
    return

def main():
    # run_feasible_exps()
    # run_trauma_exps()
    # run_runtime_exps()
    # run_pretty_paths_exps()
    run_st_pairs_exps()

if __name__ == '__main__':
    main()