import os
import networkx as nx
import numpy as np
from typing import Optional, Sequence, Union, Callable, Tuple, Dict, Any
from customtypes import Layout, Communities
import os.path as op
import sys
NETWORK_DIR = 'networks'


def write_network(G: nx.Graph,
                  network_name: str,
                  layout: Layout,
                  communities: Communities) -> None:
    G = nx.Graph(G)
    # sometimes non tuple types slip through the cracks
    if not isinstance(layout[0], tuple):
        layout = {node: (location[0], location[1]) for node, location in layout.items()}

    nx.set_node_attributes(G, layout, 'layout')
    nx.set_node_attributes(G, communities, 'community')

    nx.write_gml(G, network_name+'.txt')


def read_network(file_name: str) -> Tuple[nx.Graph, Optional[Layout], Optional[Communities]]:
    G: nx.Graph = nx.read_gml(file_name, None)  # type: ignore

    layout = nx.get_node_attributes(G, 'layout')
    if len(layout) != len(G):
        layout = None

    node_to_community = nx.get_node_attributes(G, 'community')
    if len(node_to_community) != len(G):
        node_to_community = None

    # get rid of the extraneous information
    G = nx.Graph(nx.to_numpy_array(G))

    return G, layout, node_to_community


def old_output_network(G: nx.Graph, network_name: str,
                       layout_algorithm: Optional[Union[Callable, Layout]] = None):
    """
    Save the network to a file using the deprecated format.
    """
    with open(network_name+'.txt', 'w') as f:
        f.write(f'{len(G.nodes)}\n')

        # sometimes the nodes are identified by tuples instead of just integers
        # This doesn't work with other programs in the project, so we must give each tuple
        # a unique integer ID.
        node_to_id: Dict[Any, int] = {}
        current_id = 0

        def get_id_of_node(node) -> int:
            nonlocal current_id
            if node not in node_to_id:
                node_to_id[node] = current_id
                current_id += 1
            return node_to_id[node]

        edge_lines = [f'{get_id_of_node(n0)} {get_id_of_node(n1)}\n' for n0, n1 in G.edges]
        f.writelines(edge_lines)
        # this code is just for the visualization program I made ("graph-visualizer")
        # It writes out where each of the nodes should be drawn.
        f.write('\n')
        if layout_algorithm is None:
            layout_algorithm = nx.kamada_kawai_layout
        if callable(layout_algorithm):
            layout = layout_algorithm(G)
        else:
            layout = layout_algorithm
        f.writelines(f'{node_to_id[node]} {x} {y}\n'
                     for node, (x, y) in sorted(layout.items(), key=lambda x: x[0]))
        f.write('\n')


def old_read_network_file(file_name: str) -> Tuple[np.ndarray, Optional[Layout]]:
    """Read a network in the deprecated format."""
    with open(file_name, 'r') as f:
        line = f.readline()
        shape = (int(line[:-1]), int(line[:-1]))
        matrix = np.zeros(shape, dtype='uint8')

        line = f.readline()[:-1]
        i = 1
        while len(line) > 0:
            coord = line.split(' ')
            matrix[int(coord[0]), int(coord[1])] = 1
            matrix[int(coord[1]), int(coord[0])] = 1
            line = f.readline()[:-1]
            i += 1

        rest_of_lines = tuple(map(lambda s: s.split(),
                              filter(lambda s: len(s) > 1, f.readlines())))
        layout = {int(line[0]): (float(line[1]), float(line[2]))
                  for line in rest_of_lines} if len(rest_of_lines) > 0 else None
    return matrix, layout


def get_network_name(path_string: str) -> str:
    """Return only the name portion of the path to a network."""
    return '.'.join(op.basename(path_string).split('.')[:-1])


def network_names_to_paths(network_names: Sequence[str]) -> Sequence[str]:
    """Return the paths to the named networks. Report and exit if they cannot be found."""
    network_paths = tuple(NETWORK_DIR+name+'.txt' for name in network_names)
    error_free = True
    for path in network_paths:
        if not os.path.exists(path):
            print(path, 'does not exist.', file=sys.stderr)
            error_free = False

    if not error_free:
        print('Fix errors before continuing.', file=sys.stderr)
        exit(1)

    return network_paths
