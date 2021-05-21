#!/usr/bin/python3

from typing import List, Tuple, Dict
import sys
from itertools import product

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from customtypes import Layout, NodeColors, Agent
from fileio import output_network


# make a component-gate graph
def main(argv):
    # cgg_entry_point(argv)
    # social_circles_entry_point(argv)
    output_network(nx.connected_watts_strogatz_graph(500, 4, .1), nx.circular_layout)


def cgg_entry_point(argv):
    if len(argv) < 3:
        print(f'Usage: {argv[0]} <num-big-components> <big-component-size> <gate-size>')
        return

    num_big_components = int(argv[1])
    big_component_size = int(argv[2])
    gate_size = int(argv[3])

    graph = make_complete_clique_gate_graph(num_big_components, big_component_size, gate_size)

    output_network(graph)
    nx.draw(graph, node_size=100)
    plt.show()


def social_circles_entry_point(argv):
    agents = {Agent('green', 30): 350, Agent('blue', 40): 100, Agent('purple', 50): 50}
    G, layout, _ = make_social_circles_network(agents, (500, 500))
    output_network(G, layout)


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


def make_social_circles_network(agent_type_to_quantity: Dict[Agent, int],
                                grid_size: Tuple[int, int], pbar=False,
                                force_connected=True) -> Tuple[nx.Graph, Layout, NodeColors]:
    # repeat the algorithm up to some maximum attempting to generate a connected network.
    max_tries = 100
    for _ in range(max_tries):
        agents = sorted(agent_type_to_quantity.items(),
                        key=lambda x: x[0][1], reverse=True)
        grid = np.zeros(grid_size, dtype='uint8')
        num_nodes = sum(agent_type_to_quantity.values())
        M = np.zeros((num_nodes, num_nodes), dtype='uint8')
        # place the agents with the largest reach first
        loc_to_id = {}
        current_id = 0
        for agent, quantity in agents:
            new_agents = []
            for _ in range(quantity):
                x, y = choose_empty_spot(grid)
                grid[x, y] = agent.reach
                new_agents.append((x, y))
                loc_to_id[(x, y)] = current_id
                current_id += 1
            for x, y in new_agents:
                neighbors = search_for_neighbors(grid, x, y)
                for i, j in neighbors:
                    M[loc_to_id[(x, y)], loc_to_id[(i, j)]] = 1
                    M[loc_to_id[(i, j)], loc_to_id[(x, y)]] = 1

        colors = []
        for agent, quantity in agent_type_to_quantity.items():
            colors += [agent.color]*quantity
        layout = {id_: (2*x/grid_size[0]-1, 2*y/grid_size[1]-1)
                  for (x, y), id_ in loc_to_id.items()}
        G = nx.Graph(M)
        # return the generated network if it is connected or if we don't care
        if (not force_connected) or nx.is_connected(G):
            return G, layout, colors

    # Abort execution on failure
    exit(f'Failed to generate a connected network after {max_tries} tries.')


def choose_empty_spot(grid):
    x, y = np.random.randint(grid.shape[0]), np.random.randint(grid.shape[1])
    while grid[x, y] > 0:
        x, y = np.random.randint(grid.shape[0]), np.random.randint(grid.shape[1])
    return x, y


def search_for_neighbors(grid, x, y):
    reach = grid[x, y]
    min_x = max(0, x-reach)
    max_x = min(grid.shape[0]-1, x+reach)
    min_y = max(0, y-reach)
    max_y = min(grid.shape[1]-1, y+reach)
    neighbors = {(i, j)
                 for (i, j) in product(range(min_x, max_x),
                                       range(min_y, max_y))
                 if grid[i, j] > 0 and (x, y) != (i, j)}
    return neighbors


if __name__ == '__main__':
    main(sys.argv)
