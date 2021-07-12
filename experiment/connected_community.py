from common import safe_run_trials
from sim_dynamic import Disease, simulate, FlickerBehavior, make_starting_sir
import numpy as np
from typing import Optional, Tuple, Any
from abc import ABC, abstractmethod
from customtypes import Number
import networkgen
import networkx as nx
from analyzer import COLORS, visualize_network


def poisson_entry_point():
    """Run experiments on connected community networks with a poisson degree distribution."""
    print('Running poisson connected community experiments')
    N_comm = 10  # agents per community
    num_comms = 50  # number of communities
    num_trials = 1000
    rand = np.random.default_rng(420)
    lam = 8  # this is lambda for both the inner and outer degree distributions
    configuration = PoissonConfiguration(f'Poisson: lam={lam}, {num_comms} comms, {N_comm} big',
                                         rand, lam, lam, num_comms, N_comm)
    disease = Disease(4, .2)

    safe_run_trials(configuration.name, run_connected_community_trial,
                    (configuration, disease, rand), num_trials)


def uniform_entry_point():
    """Run experiments on connected community networks with a unifrom degree distribution."""
    print('Running uniform connected community experiments')
    N_comm = 10  # agents per community
    num_comms = 50  # number of communities
    num_trials = 1000
    rand = np.random.default_rng(0)
    inner_bounds = (5, 9)
    outer_bounds = (2, 5)
    configuration = UniformConfiguration(rand, inner_bounds, outer_bounds, N_comm, num_comms)
    disease = Disease(4, .2)

    safe_run_trials(configuration.name, run_connected_community_trial,
                    (configuration, disease, rand), num_trials)


class ConnectedCommunityConfiguration(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def make_inner_degrees(self) -> np.ndarray:
        pass

    @abstractmethod
    def make_outer_degrees(self) -> np.ndarray:
        pass


class PoissonConfiguration(ConnectedCommunityConfiguration):
    def __init__(self, name: str, rand, inner_lam: Number, outer_lam: Number,
                 N_comm: int, num_communities: int) -> None:
        self._name = name
        self._inner_lam = inner_lam
        self._outer_lam = outer_lam
        self._rand = rand
        self._N_comm = N_comm
        self._num_communities = num_communities

    @property
    def name(self) -> str:
        return self._name

    def make_inner_degrees(self) -> np.ndarray:
        inner_degrees = np.round(self._rand.poisson(self._inner_lam, size=self._N_comm))
        if np.sum(inner_degrees) % 2 == 1:
            inner_degrees[np.argmin(inner_degrees)] += 1
        return inner_degrees

    def make_outer_degrees(self) -> np.ndarray:
        outer_degrees = np.round(self._rand.poisson(self._outer_lam, size=self._num_communities))
        if np.sum(outer_degrees) % 2 == 1:
            outer_degrees[np.argmin(outer_degrees)] += 1
        return outer_degrees


class UniformConfiguration(ConnectedCommunityConfiguration):
    def __init__(self, rand,
                 inner_bounds: Tuple[int, int], outer_bounds: Tuple[int, int],
                 N_comm: int, num_communities: int) -> None:
        """
        An object that stores the configuration for a connected community network.

        inner_bounds: The (lower, upper) bounds for the intracommunity degree distribution
        outer_bounds: The (lower, upper) bounds for the intercommunity degree distribution
        N_comm: The number of nodes in a community
        num_communities: The number of communities
        """
        self._name = f'UniformConfig(ib={inner_bounds}, ob={outer_bounds}'\
                     f'N_comm={N_comm}, num_comms = {num_communities})'
        self._rand = rand
        self._inner_bounds = inner_bounds
        self._outer_bounds = outer_bounds
        self._N_comm = N_comm
        self._num_communities = num_communities

    @property
    def name(self) -> str:
        return self._name

    def make_inner_degrees(self) -> np.ndarray:
        inner_degrees = self._rand.integers(self._inner_bounds[0], self._inner_bounds[1],
                                            self._N_comm)
        if np.sum(inner_degrees) % 2 == 1:
            inner_degrees[np.argmin(inner_degrees)] += 1
        return inner_degrees

    def make_outer_degrees(self) -> np.ndarray:
        outer_degrees = self._rand.integers(self._outer_bounds[0], self._outer_bounds[1],
                                            self._num_communities)
        if np.sum(outer_degrees) % 2 == 1:
            outer_degrees[np.argmin(outer_degrees)] += 1
        return outer_degrees


def run_connected_community_trial(args: Tuple[ConnectedCommunityConfiguration, Disease, Any])\
        -> Optional[Tuple[float, float]]:
    """
    args: (ConnectedCommunityConfiguration to use,
           disease to use,
           default_rng instance)
    return: (proportion of edges that flicker, the average number of remaining susceptible agents)
            or None on failure.
    """
    sim_len = 200
    sims_per_trial = 150
    configuration, disease, rand = args

    inner_degrees = configuration.make_inner_degrees()
    outer_degrees = configuration.make_outer_degrees()
    cc_results = networkgen.make_connected_community_network(inner_degrees, outer_degrees,
                                                             rand)
    # If a network couldn't be successfully generated, return None to signal the failure
    if cc_results is None:
        return None
    G, communities = cc_results
    to_flicker = {(u, v) for u, v in G.edges if communities[u] != communities[v]}
    proportion_flickering = len(to_flicker) / len(G.edges)
    M = nx.to_numpy_array(G)

    behavior = FlickerBehavior(M, to_flicker, (True, False), "Probs don't change this")
    avg_sus = np.mean([np.sum(simulate(M, make_starting_sir(len(M), 1),
                                       disease, behavior, sim_len, None, rand)[-1][0] > 0)
                       for _ in range(sims_per_trial)])

    return proportion_flickering, avg_sus


def visual_inspection():
    N_comm = 10  # agents per community
    num_comms = 50  # number of communities
    rand = np.random.default_rng(0)
    inner_bounds = (5, 9)
    outer_bounds = (2, 5)
    configuration = UniformConfiguration(rand, inner_bounds, outer_bounds, N_comm, num_comms)

    for i in range(10):
        inner_degrees = configuration.make_inner_degrees()
        outer_degrees = configuration.make_outer_degrees()
        cc_results = networkgen.make_connected_community_network(inner_degrees, outer_degrees, rand)
        if cc_results is None:
            print('Failure')
            continue
        G, communities = cc_results
        node_color = [COLORS[community] for community in communities.values()]
        visualize_network(G, None, name=str(i), block=False, node_color=node_color)  # type: ignore
    input('Done')


if __name__ == '__main__':
    visual_inspection()
