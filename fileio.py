import networkx as nx
import numpy as np
from typing import Optional, Union, Callable, Tuple
from customtypes import Layout


def output_network(G: nx.Graph, layout_algorithm: Optional[Union[Callable, Layout]] = None):
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
    if layout_algorithm is None:
        layout_algorithm = nx.kamada_kawai_layout
    if callable(layout_algorithm):
        layout = layout_algorithm(G)
    else:
        layout = layout_algorithm
    for node, coordinate in sorted(layout.items(), key=lambda x: x[0]):
        print(f'{node_to_id[node]} {coordinate[0]} {coordinate[1]}')
    print()


def read_network_file(file_name: str) -> Tuple[np.ndarray, Optional[Layout]]:
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
