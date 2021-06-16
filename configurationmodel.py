#!/usr/bin/python3
import itertools as it
from customtypes import Number
from typing import Callable, Sequence, Tuple
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import hcmioptim as ho
from analyzer import colors_from_communities


def main():
    n_steps = 5000
    N = 100
    rand = np.random.default_rng()
    # node_to_degree = np.clip(rand.normal(5, 3, N), 1, None).astype('int')
    # # All of the degrees must sum to an even number. This if block ensures that happens.
    # if np.sum(node_to_degree) % 2 == 1:
    #     node_to_degree[np.argmin(node_to_degree)] += 1
    # edge_list = np.array([node
    #                       for node, degree in enumerate(node_to_degree)
    #                       for _ in range(degree)])
    # rand.shuffle(edge_list)
    # optimizer = ho.sa.SAOptimizer(component_objective, ho.sa.make_fast_schedule(100),
    #                               make_edge_list_neighbor(), edge_list, True)
    optimizer = ho.ga.GAOptimizer(component_objective,
                                  make_next_network_gen(N, rand),
                                  make_starting_pop(20, N, rand),
                                  True)
    pbar = tqdm(range(n_steps))
    costs = np.zeros(n_steps)
    global_best = None
    for step in pbar:
        cost_to_encoding = optimizer.step()
        local_best = min(cost_to_encoding, key=lambda x: x[0])
        if global_best is None or local_best[0] < global_best[0]:
            global_best = local_best
        costs[step] = local_best[0]
        pbar.set_description('Cost: {:.3f}'.format(local_best[0]))

    G = network_from_edge_list(global_best[1])  # type: ignore
    print('Number of nodes:', len(G.nodes))
    print('Number of edges:', len(G.edges))
    print('Number of components:', len(tuple(nx.connected_components(G))))

    plt.plot(costs)
    plt.show(block=False)
    plt.figure()
    plt.hist(tuple(x[1] for x in G.degree), bins=None)
    plt.show(block=False)
    plt.figure()
    nx.draw_kamada_kawai(G, node_size=50, node_color=colors_from_communities(nx.connected_components(G)))
    plt.show(block=False)

    input('Press <enter> to exit.')


def make_clustering_objective():
    rand = np.random.default_rng()

    def clustering_objective(degrees: Sequence[int]) -> float:
        networks = (network_from_degree_sequence(degrees, make_vis_func(False), False, rand)
                    for _ in range(25))
        clustering_coefficients = tuple(nx.average_clustering(G) for G in networks)
        energy = sum(clustering_coefficients) / len(clustering_coefficients)
        return energy

    return clustering_objective


def component_objective(edge_list: np.ndarray) -> int:
    G = network_from_edge_list(edge_list)
    conn_comps = tuple(nx.connected_components(G))
    largest_component = max(conn_comps, key=len)
    bad = len(largest_component) + len(conn_comps)
    good = nx.diameter(G.subgraph(largest_component)) + len(G) + len(G.edges)  # type: ignore
    energy = bad - good
    return energy


def configuration_neighbor(degrees: Sequence[int], rand) -> Sequence[int]:
    neighbor = np.copy(degrees)
    nonzero_entries = np.where(neighbor > 0)[0]
    i, j = rand.choice(nonzero_entries, 2, replace=False)
    neighbor[i] += rand.choice((-1, 1))
    neighbor[j] += rand.choice((-1, 1))
    return neighbor


def make_edge_list_neighbor() -> Callable[[np.ndarray], np.ndarray]:
    rand = np.random.default_rng()

    def edge_list_neighbor(edge_list: np.ndarray) -> np.ndarray:
        index0 = rand.integers(0, edge_list.shape[0])
        index1 = rand.integers(0, edge_list.shape[0])

        # to eliminate self-loops check the value adjacent to index0 to make sure edge_list[index1] != that_value
        offset = 1 if index0 % 2 == 0 else -1
        while edge_list[index0+offset] == edge_list[index1]:
            index1 = rand.integers(0, edge_list.shape[0])

        new_edge_list = np.copy(edge_list)
        new_edge_list[index0], new_edge_list[index1] = new_edge_list[index1], new_edge_list[index0]
        return new_edge_list

    return edge_list_neighbor


def network_from_degree_sequence(degrees: Sequence[int], vis_func, force_good_behavior: bool, rand) -> nx.Graph:
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
            return network_from_degree_sequence(degrees, vis_func, force_good_behavior, rand)

        v = rand.choice(available_nodes)
        node_to_remaining_stubs[v] -= 1
        G.add_edge(u, v)
        vis_func(G, u, v)

    return G


def make_next_network_gen(N: int, rand):
    vertex_choices = tuple(range(N))

    def next_network_gen(degree_sequences: Sequence[Tuple[Number, np.ndarray]]) -> Tuple[np.ndarray, ...]:
        couples = ho.ga.roulette_wheel_cost_selection(degree_sequences)
        offspring = (ho.ga.single_point_crossover(*couple) for couple in couples)
        children = tuple(child for pair in offspring for child in pair)
        for i, j in it.product(range(len(children)), range(len(children[0]))):
            if rand.random() < .001:
                children[i][j] = rand.choice(vertex_choices)

        return children

    return next_network_gen


def make_starting_pop(size: int, N: int, rand) -> Tuple[np.ndarray, ...]:
    population = tuple(np.clip(rand.normal(15, 3, N), 1, None).astype('int') for _ in range(size))
    for i, edge_list in enumerate(population):
        if np.sum(edge_list) % 2 != 0:
            population[i][np.argmin(edge_list)] += 1
    return population


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
