#!/usr/bin/python3
import itertools as it
from partitioning import fluidc_partition
from sim_dynamic import Disease, PatternFlickerBehavior, make_starting_sir, simulate
from socialgood import rate_social_good
from customtypes import Number
from typing import Callable, List, Sequence, Tuple
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import hcmioptim.ga as ga
from analyzer import all_same, visualize_network, betw_centrality
from encoding_lib import edge_set_to_network, edge_list_to_network


def main():
    n_steps = 1000
    N = 100
    rand = np.random.default_rng()
    # optimizer = ga.GAOptimizer(PercolationResistanceObjective(rand, 10, edge_set_to_network),
    #                            NextNetworkGenEdgeSet(rand),
    #                            new_edge_set_pop(20, N, rand),
    #                            True, 6)
    optimizer = ga.GAOptimizer(IRNObjective(edge_set_to_network, rand),
                               NextNetworkGenEdgeSet(rand),
                               new_edge_set_pop(20, N, rand),
                               True, 1)
    pbar = tqdm(range(n_steps))
    costs = np.zeros(n_steps)
    global_best: Tuple[Number, np.ndarray] = None  # type: ignore
    for step in pbar:
        cost_to_encoding = optimizer.step()
        local_best = min(cost_to_encoding, key=lambda x: x[0])
        if global_best is None or local_best[0] < global_best[0]:
            global_best = local_best
        costs[step] = local_best[0]
        pbar.set_description('Cost: {:.3f}'.format(local_best[0]))
        if global_best[0] == 1:
            break

    G = edge_set_to_network(global_best[1])
    print('Number of nodes:', len(G.nodes))
    print('Number of edges:', len(G.edges))
    print('Number of components:', len(tuple(nx.connected_components(G))))

    plt.plot(costs)
    plt.show(block=False)
    plt.figure()
    plt.hist(tuple(x[1] for x in G.degree), bins=None)
    plt.show(block=False)
    plt.figure()
    visualize_network(G, None,
                      f'From Edge List\nCost: {global_best[0]}',
                      False, all_same, False)

    input('Press <enter> to exit.')


class PercolationResistanceObjective:
    """
    This is very interesting because it actually works pretty darn well.
    It seems to create resilient hairballs.
    """
    def __init__(self, rand, edges_to_remove: int,
                 encoding_to_network: Callable[[np.ndarray], nx.Graph]):
        self._rand = rand
        self._encoding_to_network = encoding_to_network
        self._edges_to_remove = edges_to_remove

    def __call__(self, encoding: np.ndarray) -> int:
        n_components = np.zeros(100, dtype=np.uint32)
        G = self._encoding_to_network(encoding)
        for i in range(len(n_components)):
            G1 = nx.Graph(G)
            edges = list(G1.edges)
            self._rand.shuffle(edges)
            G1.remove_edges_from(edges[:self._edges_to_remove])
            n_components[i] = nx.number_connected_components(G1)
        N = len(G)
        E = (N**2-N)//2
        return np.sum(n_components)/len(n_components) + 2*len(G.edges)/E


class IRNObjective:
    def __init__(self, encoding_to_network: Callable[[np.ndarray], nx.Graph], rand) -> None:
        self._enc_to_G = encoding_to_network
        self._rand = rand
        self._disease = Disease(4, .2)
        self._sim_len = 100
        self._n_sims = 100

    def __call__(self, encoding: np.ndarray) -> float:
        G = self._enc_to_G(encoding)
        M = nx.to_numpy_array(G)
        to_flicker = fluidc_partition(G, len(G)//20)
        flicker_behavior = PatternFlickerBehavior(M, to_flicker, (True, False), '')
        avg_sus = np.mean([np.sum(simulate(M, make_starting_sir(len(M), 1),
                                           self._disease, flicker_behavior, self._sim_len,
                                           None, self._rand)[-1][0] > 0)
                           for _ in range(self._n_sims)]) / len(M)
        cost = 2-avg_sus-rate_social_good(M)
        return cost


class ClusteringObjective:
    def __init__(self, encoding_to_network: Callable[[np.ndarray], nx.Graph],
                 diameter_weight: float):
        self._rand = np.random.default_rng()
        self._encoding_to_network = encoding_to_network
        self._diameter_weight = diameter_weight

    def __call__(self, encoding) -> float:
        G = self._encoding_to_network(encoding)
        biggest_comp = G.subgraph(max(nx.connected_components(G), key=len))
        return -nx.average_clustering(G)\
            - self._diameter_weight * nx.diameter(biggest_comp)  # type: ignore


def component_objective(edge_list: np.ndarray) -> int:
    G = edge_list_to_network(edge_list)
    conn_comps = tuple(nx.connected_components(G))
    largest_component = max(conn_comps, key=len)
    bad = len(largest_component)*2 + len(conn_comps)
    good = nx.diameter(G.subgraph(largest_component)) + len(G) + len(G.edges)  # type: ignore
    energy = bad - good
    return energy


class HighBetweenessObjective:
    def __init__(self, encoding_to_network: Callable[[np.ndarray], nx.Graph],
                 num_important_edges: int,
                 diameter_weight: float) -> None:
        self._encoding_to_network = encoding_to_network
        self._num_important_edges = num_important_edges
        self._diameter_weight = diameter_weight

    def __call__(self, encoding: np.ndarray) -> float:
        G = self._encoding_to_network(encoding)

        if not nx.is_connected(G):
            return len(G) + self._num_important_edges*2

        edge_betwenesses = sorted(nx.edge_betweenness_centrality(G).values(),  # type: ignore
                                  reverse=True)
        return -sum(edge_betwenesses[:self._num_important_edges])\
            - nx.diameter(G)*self._diameter_weight  # type: ignore


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

        # to eliminate self-loops check the value adjacent to index0
        # to make sure edge_list[index1] != that_value
        offset = 1 if index0 % 2 == 0 else -1
        while edge_list[index0+offset] == edge_list[index1]:
            index1 = rand.integers(0, edge_list.shape[0])

        new_edge_list = np.copy(edge_list)
        new_edge_list[index0], new_edge_list[index1] = new_edge_list[index1], new_edge_list[index0]
        return new_edge_list

    return edge_list_neighbor


class NextNetworkGenEdgeList:
    def __init__(self, N: int, rand):
        self._vertex_choices = tuple(range(N))
        self._N = N
        self._rand = rand

    def __call__(self, rated_pop: Sequence[Tuple[Number, np.ndarray]]) -> Tuple[np.ndarray, ...]:
        couples = ga.roulette_wheel_cost_selection(rated_pop)
        offspring = (ga.single_point_crossover(*couple) for couple in couples)
        children = tuple(child for pair in offspring for child in pair)

        for i, j in it.product(range(len(children)), range(len(children[0]))):
            if self._rand.random() < .001:
                children[i][j] = self._rand.choice(self._vertex_choices)

        return children


class NextNetworkGenEdgeSet:
    def __init__(self, rand) -> None:
        self._rand = rand

    def __call__(self, rated_pop: Sequence[Tuple[Number, np.ndarray]]) -> Tuple[np.ndarray, ...]:
        couples = ga.roulette_wheel_cost_selection(rated_pop)
        offspring = (ga.single_point_crossover(*couple) for couple in couples)
        children = tuple(child for pair in offspring for child in pair)

        for i, j in it.product(range(len(children)), range(len(children[0]))):
            if self._rand.random() < .0001:
                children[i][j] = 1 - children[i][j]

        return children


def new_edge_list_pop(population_size: int, N: int, rand) -> Tuple[np.ndarray, ...]:
    # decide on a degree for each of the nodes
    node_to_degree = np.clip(rand.normal(5, 3, N), 1, None).astype('int')
    # ensure that the sum of the degrees is even
    if np.sum(node_to_degree) % 2 != 0:
        node_to_degree[np.argmin(node_to_degree)] += 1

    # put each node's id in an array (edge list) the same amount as its degree
    base_list = np.array(tuple(node
                               for node, degree in enumerate(node_to_degree)
                               for _ in range(degree)))
    population = tuple(np.copy(base_list) for _ in range(population_size))
    # shuffle each edge list
    for edge_list in population:
        rand.shuffle(edge_list)

    return population


def new_edge_set_pop(size: int, N: int, rand) -> List[np.ndarray]:
    edge_density = .01
    E = (N**2 - N) // 2
    population = [np.array([1 if i < int(E*edge_density) else 0 for i in range(E)])
                  for _ in range(size)]
    for edge_set in population:
        rand.shuffle(edge_set)
    return population


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
