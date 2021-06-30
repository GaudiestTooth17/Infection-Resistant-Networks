import numpy as np
import networkx as nx
from typing import Sequence


def degree_sequence_to_network(degrees: Sequence[int],
                               vis_func,
                               force_good_behavior: bool,
                               rand) -> nx.Graph:
    if sum(degrees) % 2 != 0:
        raise Exception('The sum of degrees must be even.')

    node_to_remaining_stubs = dict(enumerate(degrees))
    G = nx.empty_graph(len(degrees))
    while sum(node_to_remaining_stubs.values()) > 0:
        available_nodes = tuple(filter(lambda u: node_to_remaining_stubs[u],
                                       node_to_remaining_stubs.keys()))
        u = rand.choice(available_nodes)
        node_to_remaining_stubs[u] -= 1
        available_nodes = tuple(filter(lambda v: all((node_to_remaining_stubs[v],
                                                      any((not force_good_behavior,
                                                           v != u and not G.has_edge(u, v))))),
                                       node_to_remaining_stubs.keys()))
        if len(available_nodes) == 0:
            print('Network generation failed. Restarting.')
            return degree_sequence_to_network(degrees, vis_func, force_good_behavior, rand)

        v = rand.choice(available_nodes)
        node_to_remaining_stubs[v] -= 1
        G.add_edge(u, v)
        vis_func(G, u, v)

    return G


def edge_set_to_network(edge_set: np.ndarray) -> nx.Graph:
    E = edge_set.shape[0]
    N = int(np.sqrt(2*E+.25)+.5)
    current_edge = 0
    M = np.zeros((N, N), dtype=np.int64)

    for i in range(M.shape[0]):
        for j in range(i+1, M.shape[1]):
            M[i, j] = edge_set[current_edge]
            M[j, i] = edge_set[current_edge]
            current_edge += 1

    return nx.Graph(M)


def network_to_edge_set(M: np.ndarray) -> np.ndarray:
    N = M.shape[0]
    E = (N**2 - N) // 2
    edge_set = np.zeros(E, dtype=np.int64)
    edge_ind = 0
    for i in range(M.shape[0]):
        for j in range(i+1, M.shape[1]):
            edge_set[edge_ind] = M[i, j]
            edge_ind += 1
    return edge_set
