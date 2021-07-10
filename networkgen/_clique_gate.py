from fileio import write_network
from analyzer import visualize_network
import networkx as nx
from typing import List, Tuple
from customtypes import Communities
import sys


def cgg_entry_point():
    if len(sys.argv) < 4:
        print(f'Usage: {sys.argv[0]} <num-big-components> <big-component-size> <gate-size> [name]')
        return

    num_big_components = int(sys.argv[1])
    big_component_size = int(sys.argv[2])
    gate_size = int(sys.argv[3])
    name = sys.argv[4]

    G, node_to_community = make_complete_clique_gate_network(num_big_components,
                                                             big_component_size,
                                                             gate_size)
    layout = nx.kamada_kawai_layout(G)
    print(f'Network has {len(G)} nodes.')
    print(type(layout[0]))
    write_network(G, name, layout, node_to_community)
    visualize_network(G, layout, name, block=False)
    G.remove_edges_from((u, v) for u, v in G.edges if node_to_community[u] != node_to_community[v])
    visualize_network(G, layout, name+' partitioned')


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


def make_complete_clique_gate_network(num_big_components: int,
                                      big_component_size: int,
                                      gate_size: int) -> Tuple[nx.Graph, Communities]:
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
                                             len(gate_node_ids)*gate_size
                                             + num_big_components*big_component_size,
                                             big_component_size))
    big_comps = [nx.complete_graph(node_ids) for node_ids in component_node_ids]

    # put the disparate components into the same network
    master_graph = union_components(gates + big_comps)  # type: ignore

    # insert gates in between components
    current_gate_ind = 0
    for comp_ind, src_comp in enumerate(big_comps[:-1]):
        for dest_comp in big_comps[comp_ind+1:]:
            gate_nodes = list(gates[current_gate_ind].nodes())  # type: ignore
            current_gate_ind += 1
            # Add edges to src_comp.
            # The loop assumes that there are fewer or equal nodes in half
            # the gate than in each component
            src_nodes = list(src_comp.nodes())  # type: ignore
            for i, node in enumerate(gate_nodes[:len(gate_nodes)//2]):
                master_graph.add_edge(node, src_nodes[i])
            # add edges to dest_cold_read_network_fileomp
            dest_nodes = list(dest_comp.nodes())  # type: ignore
            for i, node in enumerate(gate_nodes[len(gate_nodes)//2:]):
                master_graph.add_edge(node, dest_nodes[i])

    node_to_community_id = {node: comm_id
                            for comm_id, sub_graph in enumerate(gates+big_comps)
                            for node in sub_graph}
    return master_graph, node_to_community_id
