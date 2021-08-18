from customtypes import Number
from network import Network
import numpy as np
import networkx as nx
from typing import Sequence, Tuple


def degree_sequence_to_network(degrees: Sequence[int],
                               vis_func,
                               force_good_behavior: bool,
                               rand) -> nx.Graph:
    if sum(degrees) % 2 != 0:
        raise Exception('The sum of degrees must be even.')

    node_to_remaining_stubs = dict(enumerate(degrees))
    G: nx.Graph = nx.empty_graph(len(degrees))  # type: ignore
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


def edge_set_to_network(edge_set: np.ndarray) -> Network:
    E = edge_set.shape[0]
    N = int(np.sqrt(2*E+.25)+.5)
    current_edge = 0
    M = np.zeros((N, N), dtype=np.int64)

    for i in range(M.shape[0]):
        for j in range(i+1, M.shape[1]):
            M[i, j] = edge_set[current_edge]
            M[j, i] = edge_set[current_edge]
            current_edge += 1

    return Network(M)


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


def edge_list_to_network(edge_list: np.ndarray) -> nx.Graph:
    """
    edge_list is the concrete implementation of the thing created by a degree configuration.
    If a node has degree n, that node's integer ID will appear n times in edge_list. Edges are
    created by putting the values in groups of two starting at index 0 and proceeding to the end.
    In this way, edge_list can be viewed as a List[Tuple[int, int]].
    """
    N = np.max(edge_list)
    G: nx.Graph = nx.empty_graph(N)  # type: ignore
    G.add_edges_from(((edge_list[i], edge_list[i+1])
                      for i in range(0, len(edge_list), 2)))
    return G


def calc_edge_set_population_diversity(rated_population: Sequence[Tuple[Number, np.ndarray]])\
        -> float:
    """
    I think my approach is flawed because the diversity is always so low, even
    when the population is randomly generated.
    """
    diversity = 0
    genotype_len = len(rated_population[0][1])
    pop_size = len(rated_population)
    n_unique_pairings = (pop_size**2 - pop_size) // 2
    normalization_quantity = n_unique_pairings * genotype_len
    # first set diversity equal to how undiverse the population is
    for i in range(len(rated_population)):
        for j in range(i+1, len(rated_population)):
            # add 1 for each time an index is the same
            diversity += np.sum(rated_population[i][1] == rated_population[j][1])
    # normalize
    diversity /= normalization_quantity
    # subtract from 1 to change from uniformity to diversity
    diversity = 1 - diversity
    return diversity


def calc_generic_population_diversity(rated_population: Sequence[Tuple[Number, np.ndarray]])\
        -> float:
    """
    Take a general approach to calculating diversity that works with any sort of genotype.

    Find the number of unique genotypes and divide this by the number of genotypes.
    """
    n_unique = len(set(cost_to_genotype[1].tobytes() for cost_to_genotype in rated_population))
    return n_unique / len(rated_population)
