import sys
sys.path.append('')
from typing import Sequence, Tuple
from networkgen import make_affiliation_network
import krug.ga as ga
import encoding_lib as lib
from tqdm import tqdm
import matplotlib.pyplot as plt
from network import Network
import numpy as np
from scipy.stats import entropy
import networkx as nx
import itertools as it
from multiprocessing import Pool


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


def test_consistancy(N: int, group_membership: np.ndarray, id_: int):
    objective = AffiliationObjective(N, .01, .3, 100, np.random.default_rng(66))
    _, edge_densities, clustering_coefficients = objective.run(group_membership)
    base_name = f'N={N}_{id_}'
    for dist_name, distribution in zip(('Edge Densities', 'Clustering Coefficients'),
                                       (edge_densities, clustering_coefficients)):
        plot_name = f'{dist_name} {base_name}\nEntropy: {calc_entropy(distribution, 1000):.4f}'
        plt.figure()
        plt.title(plot_name)
        plt.boxplot(distribution)
        plt.savefig(f'results/{plot_name}.png', format='png')


def test_consistancy_wrapper(args):
    return test_consistancy(*args)


def test_consistancies_of_random_arrays():
    rng = np.random.default_rng(69)
    list_of_args = [(1000, rng.random(100)*.1, i) for i in range(10)]
    with Pool(5) as p:
        p.map(test_consistancy_wrapper, list_of_args, 2)


def calc_entropy(a: np.ndarray, bins: int) -> float:
    hist, _ = np.histogram(a, bins=bins)
    return entropy(hist)


if __name__ == '__main__':
    test_consistancies_of_random_arrays()
