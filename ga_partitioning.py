#!/usr/bin/python3

from analyzer import visualize_graph
from hcmioptim import ga
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Sequence, Tuple
import sys
import itertools as it
import label_partitioning
from tqdm.std import tqdm
from customtypes import Number
from fileio import read_network_file


def main():
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <network> <num components>')
        return

    M, layout = read_network_file(sys.argv[1])
    G = nx.Graph(M)
    rand = np.random.default_rng(0)
    # objective = PartitioningObjective(G)
    # optimizer = ga.GAOptimizer(objective,
    #                            NextEdgesToRm(rand),
    #                            new_to_rm_pop(len(G.edges), 20, rand),
    #                            True, 1)  # it's like 4x faster with only one core
    # optimizer = ga.GAOptimizer(ChakrabortySatoObjective(G),
    #                            NextChakrabortySatoGen(rand, G),
    #                            new_chakraborty_sato_pop(rand, G, 50),
    #                            True, 5)
    n_comps = int(sys.argv[2])
    n_labels = n_comps
    print(f'Searching for {n_comps} components.')
    optimizer = ga.GAOptimizer(LabelObjective(G, n_comps),
                               NextLabelGen(n_labels, rand),
                               new_label_pop(rand, len(G), 50, n_labels),
                               True, 2)

    n_steps = 200
    pbar = tqdm(range(n_steps))
    costs = np.zeros(n_steps)
    diversities = np.zeros(n_steps)
    global_best: Tuple = None  # type: ignore
    for step in pbar:
        cost_to_encoding = optimizer.step()
        local_best = min(cost_to_encoding, key=lambda x: x[0])
        if global_best is None or local_best[0] < global_best[0]:
            global_best = local_best
        costs[step] = local_best[0]
        diversities[step] = len({tuple(ce[1]) for ce in cost_to_encoding}) / len(cost_to_encoding)
        pbar.set_description('Cost: {:.3f}'.format(local_best[0]))

    # partitioned = objective.partition(global_best[1])
    # partitioned = chakraborty_sato_partition(G, global_best[1])
    to_remove = label_partitioning.partition(G, global_best[1])
    partitioned = nx.Graph(G)
    partitioned.remove_edges_from(to_remove)
    print('Cost:', global_best[0])

    plt.title('Diversity')
    plt.plot(diversities)
    plt.figure()
    plt.title('Cost')
    plt.plot(costs)
    plt.figure()
    visualize_graph(partitioned, layout, 'Partitioned via Label GA')


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


class ChakrabortySatoObjective:
    def __init__(self, G: nx.Graph) -> None:
        """The objective function for community detection as described by Chakraborty and Sato."""
        self._E = len(G.edges)
        self._M = nx.to_numpy_array(G)
        self._nodes = tuple(G.nodes)
        self._k: Dict = nx.degree(G)  # type: ignore

    def __call__(self, encoding: np.ndarray) -> float:
        """
        Rate the encoding based on how well it clusters.

        Encodings encode a list of N edges. One end of the edge is the location in the array.
        The other is the value.
        """
        cluster_forest = nx.Graph(tuple((u, v) for u, v in enumerate(encoding)))
        clusters = tuple(nx.connected_components(cluster_forest))
        # import pdb; pdb.set_trace()

        def lookup_cluster(node):
            for cluster in clusters:
                if node in cluster:
                    return cluster
            raise Exception(f'Could not find cluster for {node}.')
        node_to_cluster = {node: lookup_cluster(node) for node in self._nodes}

        return -(1/(2*self._E)) * np.sum(tuple((self._M[i, j] - (self._k[i]*self._k[j])/(2*self._E))
                                         * self._delta(node_to_cluster[i], node_to_cluster[j])
                                               for i, j in it.product(self._nodes, self._nodes)))

    @staticmethod
    def _delta(cluster_i, cluster_j) -> int:
        return cluster_i == cluster_j


class NextChakrabortySatoGen:
    def __init__(self, rand, G: nx.Graph) -> None:
        self._rand = rand
        self._neighbors = {node: np.array(tuple(nx.neighbors(G, node))) for node in G}

    def __call__(self, rated_pop: Sequence[Tuple[Number, np.ndarray]]) -> Tuple[np.ndarray, ...]:
        couples = ga.tournament_selection(rated_pop)
        offspring = (ga.single_point_crossover(*couple) for couple in couples)
        children = tuple(child for pair in offspring for child in pair)

        for i, j in it.product(range(len(children)), range(len(children[0]))):
            if self._rand.random() < .001:
                children[i][j] = self._rand.choice(self._neighbors[j])

        return children


def new_chakraborty_sato_pop(rand, G: nx.Graph, size: int) -> Tuple[np.ndarray, ...]:
    neighbors = {node: tuple(nx.neighbors(G, node)) for node in G}

    def make_encoding():
        encoding = np.zeros(len(G), dtype=np.int64)
        for i in range(encoding.shape[0]):
            encoding[i] = rand.choice(neighbors[i])
        return encoding
    return tuple(make_encoding() for _ in range(size))


def chakraborty_sato_partition(G: nx.Graph, encoding: np.ndarray) -> nx.Graph:
    node_to_cluster = {}
    current_cluster = 0
    for u, v in enumerate(encoding):
        u_registered = u in node_to_cluster
        v_registered = v in node_to_cluster
        if not u_registered and not v_registered:
            node_to_cluster[u] = current_cluster
            node_to_cluster[v] = current_cluster
            current_cluster += 1
        elif not u_registered:
            node_to_cluster[u] = node_to_cluster[v]
        elif not v_registered:
            node_to_cluster[v] = node_to_cluster[u]

    edges_to_remove = filter(lambda edge: node_to_cluster[edge[0]] != node_to_cluster[edge[1]], G.edges)
    partitioned = nx.Graph(G)
    partitioned.remove_edges_from(edges_to_remove)
    return partitioned


class LabelObjective:
    def __init__(self, G: nx.Graph, num_communities: int) -> None:
        """
        An objective for finding good label partitions. The goal is to find a faster way
        to partition than Girvan-Newman that still yields good results.
        """
        self._G = G
        self._num_communities = num_communities

    def __call__(self, encoding: np.ndarray) -> Number:
        edges_to_remove = label_partitioning.partition(self._G, encoding)
        partitioned = nx.Graph(self._G)
        partitioned.remove_edges_from(edges_to_remove)
        communities = tuple(nx.connected_components(partitioned))

        cost = self._num_communities*2*np.abs(len(communities) - self._num_communities)\
            + len(max(communities, key=len)) - len(min(communities, key=len))

        return cost


class NextLabelGen:
    def __init__(self, num_labels: int, rand) -> None:
        self._labels = tuple(range(num_labels))
        self._rand = rand

    def __call__(self, rated_pop: Sequence[Tuple[Number, np.ndarray]]) -> Tuple[np.ndarray, ...]:
        couples = ga.roulette_wheel_cost_selection(rated_pop)
        offspring = (ga.single_point_crossover(*couple) for couple in couples)
        children = tuple(it.chain(*offspring))

        for i, j in it.product(range(len(children)), range(len(children[0]))):
            if self._rand.random() < .01:
                children[i][j] = self._rand.choice(self._labels)

        return children


def new_label_pop(rand, N: int, pop_size: int, n_labels: int) -> Tuple[np.ndarray, ...]:
    labels = range(n_labels)
    population = tuple(rand.choice(labels, size=N) for _ in range(pop_size))
    return population


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nGood bye.')
    except EOFError:
        print('\nGood bye.')
