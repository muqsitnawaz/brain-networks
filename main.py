import matplotlib.pyplot as plt
import matplotlib.animation
import pylab
import networkx as nx
import threading
import time

# Input
# n, m where n and m are dimension of the grid
# Output
# Returns a networkx graph object
def setup_graph(n, m):
    G = nx.grid_2d_graph(n, m)
    return G

# Input
# G networkx graph object
# Output
# Displays the graph in matplotlib window
def show_graph(G, ion=False):
    G = nx.convert_node_labels_to_integers(G, ordering = 'sorted')
    pos=nx.spring_layout(G)
    nx.draw(G,pos,font_size=16,with_labels=True)
    nx.set_edge_attributes(G, 1, 'capacity')
    nx.set_edge_attributes(G, 0, 'flow')
    edge_labels=dict([((u,v,),str(d['flow'])+'/'+str(d['capacity'])) for u,v,d in G.edges(data=True)])
    nx.draw_networkx_edge_labels(G,pos,font_size=8,edge_labels=edge_labels)
    if ion == True:
        plt.ion()
    plt.show()

# Input
# G network x graph object
# Output
# Display the graph in current matplotlib window
def update_graph(G):
    plt.gcf().clear()
    show_graph(G)

# Input
# G network x graph object
# Output
# Update graph according to user input
def run(G):
    while True:
        prompt = "1: Set k value\n2: Delete a region\n3:"
        inp = input(prompt)

        if inp == "2":
            update_graph(G)


def main():
    # dim = eval(input('dimensions? '))
    dim = 5
    G = setup_graph(dim, dim)
    show_graph(G, ion=True)
    run(G)

if __name__ == '__main__':
    main()