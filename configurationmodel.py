#!/usr/bin/python3
from customtypes import Number
from typing import Callable, Sequence
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import hcmioptim as ho
from analyzer import colors_from_communities


def main():
    n_steps = 1000
    N = 100
    rand = np.random.default_rng()
    node_to_degree = np.clip(rand.normal(5, 3, N), 1, None).astype('int')
    # All of the degrees must sum to an even number. This if block ensures that happens.
    if np.sum(node_to_degree) % 2 == 1:
        node_to_degree[np.argmin(node_to_degree)] += 1
    edge_list = np.array([node
                          for node, degree in enumerate(node_to_degree)
                          for _ in range(degree)])
    rand.shuffle(edge_list)
    optimizer = ho.sa.SAOptimizer(component_objective, ho.sa.make_fast_schedule(100),
                                  edge_list_neighbor, edge_list, True)
    pbar = tqdm(range(n_steps))
    energies = np.zeros(n_steps)
    for step in pbar:
        edge_list, energy = optimizer.step()
        energies[step] = energy
        pbar.set_description('Energy: {:.5f}'.format(energy))

    plt.plot(energies)
    plt.show(block=False)
    plt.figure()
    plt.hist(edge_list, bins=(max(edge_list) - min(edge_list))//4)
    plt.show(block=False)
    G = network_from_edge_list(edge_list)  # type: ignore
    plt.figure()
    nx.draw_kamada_kawai(G, node_size=50, node_color=colors_from_communities(nx.connected_components(G)))
    plt.show(block=False)

    input('Press <enter> to exit.')


def clustering_objective(degrees: Sequence[int]) -> float:
    networks = (network_from_degree_sequence(degrees, make_vis_func(False), False)
                for _ in range(25))
    clustering_coefficients = tuple(nx.average_clustering(G) for G in networks)
    energy = sum(clustering_coefficients) / len(clustering_coefficients)
    return energy


def component_objective(edge_list: np.ndarray) -> float:
    G = network_from_edge_list(edge_list)
    largest_component = max(nx.connected_components(G), key=len)
    energy = len(largest_component) - nx.diameter(G.subgraph(largest_component))  # type: ignore
    return energy


def configuration_neighbor(degrees: Sequence[int]) -> Sequence[int]:
    neighbor = np.copy(degrees)
    nonzero_entries = np.where(neighbor > 0)[0]
    i, j = np.random.choice(nonzero_entries, 2, replace=False)
    neighbor[i] += np.random.choice((-1, 1))
    neighbor[j] += np.random.choice((-1, 1))
    return neighbor


def edge_list_neighbor(edge_list: np.ndarray) -> np.ndarray:
    index0 = np.random.randint(0, edge_list.shape[0])
    index1 = np.random.randint(0, edge_list.shape[0])

    # to eliminate self-loops check the value adjacent to index0 to make sure edge_list[index1] != that_value
    offset = 1 if index0 % 2 == 0 else -1
    while edge_list[index0+offset] == edge_list[index1]:
        index1 = np.random.randint(0, edge_list.shape[0])

    new_edge_list = np.copy(edge_list)
    new_edge_list[index0], new_edge_list[index1] = new_edge_list[index1], new_edge_list[index0]
    return new_edge_list


def network_from_degree_sequence(degrees: Sequence[int], vis_func, force_good_behavior: bool) -> nx.Graph:
    if sum(degrees) % 2 != 0:
        raise Exception('The sum of degrees must be even.')

    node_to_remaining_stubs = dict(enumerate(degrees))
    G = nx.empty_graph(len(degrees))
    while sum(node_to_remaining_stubs.values()) > 0:
        available_nodes = tuple(filter(lambda u: node_to_remaining_stubs[u],
                                       node_to_remaining_stubs.keys()))
        u = np.random.choice(available_nodes)
        node_to_remaining_stubs[u] -= 1
        available_nodes = tuple(filter(lambda v: all((node_to_remaining_stubs[v],
                                                      any((not force_good_behavior,
                                                           v != u and not G.has_edge(u, v))))),
                                       node_to_remaining_stubs.keys()))
        if len(available_nodes) == 0:
            print('Network generation failed. Restarting.')
            return network_from_degree_sequence(degrees, vis_func, force_good_behavior)

        v = np.random.choice(available_nodes)
        node_to_remaining_stubs[v] -= 1
        G.add_edge(u, v)
        vis_func(G, u, v)

    return G


def network_from_edge_list(edge_list: np.ndarray) -> nx.Graph:
    """
    edge_list is the concrete implementation of the thing created by a degree configuration.
    If a node has degree n, that node's integer ID will appear n times in edge_list. Edges are
    created by putting the values in groups of two starting at index 0 and proceeding to the end.
    In this way, edge_list can be viewed as a List[Tuple[int, int]].
    """
    N = np.max(edge_list)
    G = nx.empty_graph(N)
    G.add_edges_from(((edge_list[i], edge_list[i+1])
                      for i in range(0, len(edge_list), 2)))
    return G


def make_vis_func(visualize: bool) -> Callable[[nx.Graph, int, int], None]:
    layout = None

    def do_vis(G: nx.Graph, u: int, v: int) -> None:
        nonlocal layout
        if layout is None:
            layout = nx.kamada_kawai_layout(G)
        nx.draw_networkx(G, with_labels=False, pos=layout,
                         node_color=tuple('green' if n in (u, v) else 'blue' for n in G.nodes))
        plt.pause(.5)  # type: ignore
        plt.clf()

    def dont_vis(G: nx.Graph, u: int, v: int) -> None:
        return None

    return do_vis if visualize else dont_vis


if __name__ == '__main__':
    try:
        main()
    except EOFError:
        print('\nGoodbye.')
    except KeyboardInterrupt:
        print('\nGoodbye.')
