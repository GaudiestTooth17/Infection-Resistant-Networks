#!/usr/bin/python3

from typing import List

import networkx as nx
import matplotlib.pyplot as plt
# import sys


# make a component-gate graph
def main(argv):
    if len(argv) < 3:
        print(f'Usage: {argv[0]} <num-big-components> <big-component-size> <gate-size>')
        return

    num_big_components = int(argv[1])
    big_component_size = int(argv[2])
    gate_size = int(argv[3])

    graph = make_complete_clique_gate_graph(num_big_components, big_component_size, gate_size)

    output_graph(graph)
    # nx.draw(graph)
    # plt.show()


def union_components(components: List[nx.Graph]) -> nx.Graph:
    """
    :param components: If the node id's are not unique, some nodes will get overwritten
    :return: the union of components
    """
    master_graph = nx.Graph()
    for comp in components:
        master_graph.add_nodes_from(comp.nodes())
        master_graph.add_edges_from(comp.edges())
    return master_graph


def make_complete_clique_gate_graph(num_big_components, big_component_size, gate_size):
    """
    A clique-gate graph is made up of several cliques that are psuedo nodes. The pseudo edges
    that connect them are smaller cliques (gates). Half the nodes in the gate have an edge into
    one clique and the other half are connected to the other clique.
    """
    # this splits up the list of valid ids into sublists the same size as gate_size
    gate_node_ids = [range(start, start+gate_size)
                     for start in range(0, sum(range(num_big_components))*gate_size, gate_size)]
    gates = [nx.complete_graph(node_ids) for node_ids in gate_node_ids]
    # start numbering the nodes in the big componenets at the first int not used by the gates
    component_node_ids = (range(start, start+big_component_size)
                          for start in range(len(gate_node_ids)*gate_size,
                                             len(gate_node_ids)*gate_size+num_big_components*big_component_size,
                                             big_component_size))
    big_comps = [nx.complete_graph(node_ids) for node_ids in component_node_ids]

    # union the disparate components
    master_graph = union_components(gates + big_comps)

    # insert gates in between components
    current_gate_ind = 0
    for comp_ind, src_comp in enumerate(big_comps[:-1]):
        for dest_comp in big_comps[comp_ind+1:]:
            gate_nodes = list(gates[current_gate_ind].nodes())
            current_gate_ind += 1
            # Add edges to src_comp.
            # The loop assumes that there are fewer or equal nodes in half the gate than in each component
            src_nodes = list(src_comp.nodes())
            for i, node in enumerate(gate_nodes[:len(gate_nodes)//2]):
                master_graph.add_edge(node, src_nodes[i])
            # add edges to dest_comp
            dest_nodes = list(dest_comp.nodes())
            for i, node in enumerate(gate_nodes[len(gate_nodes)//2:]):
                master_graph.add_edge(node, dest_nodes[i])

    return master_graph


def make_caveman_graph(num_cliques, clique_size) -> nx.Graph:
    """
    Make a graph consisting of cliques with some number of edges between cliques
    :param clique_size: number of nodes in each clique
    :param num_cliques: how many social circles to make
    """
    return nx.connected_caveman_graph(num_cliques, clique_size)


def output_graph(G: nx.Graph, layout_algorithm=None):
    """
    print the graph to stdout using the typical representation
    """
    print(len(G.nodes))

    # sometimes the nodes are identified by tuples instead of just integers
    # This doesn't work with other programs in the project, so we must give each tuple
    # a unique integer ID.
    node_to_id = {}
    current_id = 0
    for e in G.edges:
        n0, n1 = e[0], e[1]
        if n0 not in node_to_id:
            node_to_id[n0] = current_id
            current_id += 1
        if n1 not in node_to_id:
            node_to_id[n1] = current_id
            current_id += 1
        print(f'{node_to_id[n0]} {node_to_id[n1]}')
    # this code is just for the visualization program I made ("graph-visualizer")
    # It writes out where each of the nodes should be drawn.
    print()
    layout = nx.kamada_kawai_layout(G)
    for node, coordinate in layout.items():
        print(f'{node_to_id[node]} {coordinate[0]} {coordinate[1]}')
    print()


if __name__ == '__main__':
    G = nx.hexagonal_lattice_graph(15, 15)
    # print(len(G.nodes))
    # nx.draw_spectral(G, node_size=10)
    plt.show()
    output_graph(G, nx.spectral_layout)
    # main(sys.argv)
