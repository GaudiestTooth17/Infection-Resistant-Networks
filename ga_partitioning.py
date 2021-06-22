#!/usr/bin/python3

from analyzer import all_same, visualize_graph
from hcmioptim import ga
import networkx as nx
import numpy as np
from typing import Sequence, Tuple
import sys

from tqdm.std import tqdm
from customtypes import Number
import itertools as it
from fileio import read_network_file


def main():
    """
    Initial testing indicates that this is a REALLY good community detection algorithm!!
    It didn't do great on the elitist network, but perhaps with more time it could get better results.
    It was really crappy on Watts-Strogatz.
    """
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <network>')
        return

    M, layout = read_network_file(sys.argv[1])
    G = nx.Graph(M)
    rand = np.random.default_rng()
    objective = PartitioningObjective(G)
    optimizer = ga.GAOptimizer(objective,
                               NextEdgesToRm(rand),
                               new_to_rm_pop(len(G.edges), 20, rand),
                               True, 1)  # it's like 4x faster with only one core

    n_steps = 1000
    pbar = tqdm(range(n_steps))
    costs = np.zeros(n_steps)
    global_best = None
    for step in pbar:
        cost_to_encoding = optimizer.step()
        local_best = min(cost_to_encoding, key=lambda x: x[0])
        if global_best is None or local_best[0] < global_best[0]:
            global_best = local_best
        costs[step] = local_best[0]
        pbar.set_description('Cost: {}'.format(local_best[0]))

    partitioned = objective.partition(global_best[1])
    print('Cost:', global_best[0])

    visualize_graph(partitioned, layout, 'Partitioned via GA',
                    False, all_same, False)
    input('Press <enter> to exit.')


class PartitioningObjective:
    def __init__(self, G: nx.Graph) -> None:
        self._ind_to_edge = dict(enumerate(G.edges))
        self._partition_weight = len(self._ind_to_edge)*2
        self._M = nx.to_numpy_array(G)

    def _encoding_to_adj_matrix(self, encoding: np.ndarray) -> np.ndarray:
        enc_M = np.zeros(self._M.shape, dtype=self._M.dtype)
        for i, val in enumerate(encoding):
            if val > 0:
                u, v = self._ind_to_edge[i]
                enc_M[u, v] = val
                enc_M[v, u] = val
        return enc_M

    def partition(self, encoding: np.ndarray) -> np.ndarray:
        enc_M = self._encoding_to_adj_matrix(encoding)
        complement = self._M - enc_M
        return complement

    def __call__(self, encoding: np.ndarray) -> int:
        G = nx.Graph(self.partition(encoding))
        comps = tuple(nx.connected_components(G))
        if len(comps) == 1:
            return len(encoding)-np.sum(encoding)
        # largest_comp = len(max(comps, key=len))
        smallest_comp = len(min(comps, key=len))
        # think about a way to discourage little pieces from getting detached
        # perhaps adding k*|singletons|?
        cost = -len(comps)*self._partition_weight + np.sum(encoding) - smallest_comp
        return cost


class NextEdgesToRm:
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


def new_to_rm_pop(edges: int, size: int, rand) -> Tuple[np.ndarray, ...]:
    population = tuple(rand.integers(0, 2, edges) for _ in range(size))
    return population


if __name__ == '__main__':
    main()
