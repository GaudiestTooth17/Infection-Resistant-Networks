import sys
from typing import Sequence, Tuple
import numpy as np
import networkx as nx
import itertools as it
import krug.ga as ga
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import entropy
sys.path.append('')
from network import Network
import encoding_lib as lib


def calc_entropy(a: np.ndarray, bins: int) -> float:
    hist, _ = np.histogram(a, bins=bins)
    return entropy(hist)


def flatten(sequence):
    return tuple(it.chain(*sequence))


def make_affiliation_network(group_to_membership_percentage: Sequence[float],
                             N: int, rng) -> Network:
    """
    Nodes in association networks are connected if they belong to at least one common group.
    To generate an association network, first a bipartite network is formed with one set of
    nodes being the groups and the other the agents. The next step is to add an edge between
    all agents that share membership in at least one group. Finally, the group nodes and any
    edges attached to them are removed.

    group_to_membership_percentage: each index is associated with a group's ID and each value
                                    is what percentage of agents belong to that group
    N: The number of agents in the network
    rng: an np.random.default_rng instance
    """
    agents = tuple(range(N))
    group_memberships = [rng.choice(agents, size=int(np.round(N*perc)), replace=False)
                         for perc in group_to_membership_percentage]
    edges = flatten(tuple(it.combinations(membership, 2)) for membership in group_memberships)
    # If the graph is construct solely from the edge list, some nodes might be left out.
    # So, construct an empty graph and then add the edges
    G: nx.Graph = nx.empty_graph(N)
    G.add_edges_from(edges)
    return Network(G)


def evolve_affiliation_network(N: int, target_edge_density: float,
                               target_clustering_coefficient: float) -> Network:
    # constants
    n_trials = 25
    n_groups = 25
    pop_size = 20
    n_steps = 15

    # objects and such for optimization
    rng = np.random.default_rng(501)
    objective = AffiliationObjective(N, target_edge_density, target_clustering_coefficient,
                                     n_trials, rng)
    next_gen = NextGenGroupMemberships(rng, .01)
    optimizer = ga.GAOptimizer(objective, next_gen,
                               new_membership_population(n_groups, pop_size, rng),
                               False, 4)

    # optimization loop
    pbar = tqdm(range(n_steps))
    global_best: Tuple[float, np.ndarray] = None  # type: ignore
    costs = np.zeros(n_steps)
    diversities = np.zeros(n_steps)
    for step in pbar:
        cost_to_encoding = optimizer.step()
        local_best = min(cost_to_encoding, key=lambda x: x[0])
        if global_best is None or local_best[0] < global_best[0]:
            global_best = local_best
        costs[step] = local_best[0]
        diversities[step] = lib.calc_float_pop_diversity(cost_to_encoding)
        pbar.set_description(f'Cost: {local_best[0]:.3f} Diversity: {diversities[step]:.3f}')
        # pbar.set_description(f'Cost: {local_best[0]:.3f}')
    print(f'Total cache hits: {optimizer.num_cache_hits}')

    # show cost and diversity over time
    print(global_best[1])
    plt.title('Cost')
    plt.plot(costs)
    plt.show(block=False)
    plt.figure()
    plt.title('Diversity')
    plt.plot(diversities)
    plt.show()

    return make_affiliation_network(global_best[1], N, rng)  # type: ignore


class AffiliationObjective:
    def __init__(self, N: int, target_edge_density: float,
                 target_clustering_coefficient: float,
                 n_trials: int, rng):
        """
        Take the average clustering coefficient and edge density from n_trials
        Networks with N nodes and compare those to the provided targets.
        """
        self._N = N
        self._max_E = (N**2 - N) // 2  # max possible edges
        self._target_edge_density = target_edge_density
        self._target_clustering_coefficient = target_clustering_coefficient
        self._n_trials = n_trials
        self._rng = rng

    def __call__(self, group_to_membership_percentage: np.ndarray) -> float:
        """
        The encoding this function takes in is fed directly into make_affiliation_network
        """
        return self.run(group_to_membership_percentage)[0]

    def run(self, group_to_membership_percentage: np.ndarray)\
            -> Tuple[float, np.ndarray, np.ndarray]:
        edge_densities = np.zeros(self._n_trials)
        clustering_coeffs = np.zeros(self._n_trials)
        for trial in range(self._n_trials):
            net: Network = make_affiliation_network(group_to_membership_percentage,  # type: ignore
                                                    self._N, self._rng)
            edge_densities[trial] = net.E / self._max_E
            clustering_coeffs[trial] = nx.average_clustering(net.G)

        avg_edge_density = np.average(edge_densities)
        avg_cc = np.average(clustering_coeffs)

        return ((np.abs(avg_edge_density - self._target_edge_density)
                + np.abs(avg_cc - self._target_clustering_coefficient)),
                edge_densities, clustering_coeffs)


class NextGenGroupMemberships:
    def __init__(self, rng, mutation_prob: float):
        self._rng = rng
        self._mutation_prob = mutation_prob
        self._mutation_amount = .001

    def __call__(self, rated_pop: Sequence[Tuple[float, np.ndarray]]) -> Sequence[np.ndarray]:
        # cross over
        couples = ga.roulette_wheel_rank_selection(rated_pop, self._rng)
        children = tuple(it.chain(*(ga.single_point_crossover(*couple)
                                    for couple in couples)))
        do_mutation = self._rng.integers(0, 2, len(children)*len(children[0]))
        # mutation
        for i, j in it.product(range(len(children)), range(len(children[0]))):
            if do_mutation[i*len(children)+j] < 0:
                children[i][j] += self._mutation_amount * self._rng.choice((-1, 1))

        # make sure there are no out of bounds values
        for child in children:
            np.clip(child, 0, 1, child)
        return children


def new_membership_population(n_groups: int, pop_size: int, rng) -> Sequence[np.ndarray]:
    return tuple(rng.random(n_groups) for _ in range(pop_size))


def test_consistancy():
    # TODO: find the entropy for random group memberships that only contain low values
    group_membership = np.array([0.2337656, 0.30522161, 0.40644384, 0.14117652,
                                 0.21550407, 0.16917761, 0.29178833, 0.1564601,
                                 0.44323018, 0.70752711, 0.34885096, 0.55909792,
                                 0.1700208, 0.22228515, 0.36972174, 0.38646901,
                                 0.10031864, 0.29303319, 0.51209777, 0.20841458,
                                 0.41822691, 0.23562407, 0.3091762, 0.37521886,
                                 0.2365973])
    objective = AffiliationObjective(100, .01, .3, 1000, np.random.default_rng(66))
    _, edge_densities, clustering_coefficients = objective.run(group_membership)
    for name, distribution in zip(('Edge Densities', 'Clustering Coefficients'),
                                  (edge_densities, clustering_coefficients)):
        plt.figure()
        plt.title(name + f'\nEntropy: {calc_entropy(distribution, 1000)}')
        plt.boxplot(distribution)
    plt.show()


if __name__ == '__main__':
    # evolve_affiliation_network(100, .01, .3)
    test_consistancy()
