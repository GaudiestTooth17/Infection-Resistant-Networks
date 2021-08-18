from common import safe_run_trials
from sim_dynamic import Disease, simulate, StaticFlickerBehavior, make_starting_sir
import numpy as np
from typing import Optional, Tuple, Any
from abc import ABC, abstractmethod
from customtypes import Number
import networkgen
from analysis import COLORS, visualize_network
import sys
sys.path.append('')
from socialgood import rate_social_good


def poisson_entry_point():
    """Run experiments on connected community networks with a poisson degree distribution."""
    print('Running poisson connected community experiments')
    N_comm = 10  # agents per community
    num_comms = 50  # number of communities
    num_trials = 100
    rand = np.random.default_rng(420)
    lam = 8  # this is lambda for both the inner and outer degree distributions
    configuration = PoissonConfiguration(rand, lam, lam, num_comms, N_comm)
    disease = Disease(4, .2)

    safe_run_trials(configuration.name, run_connected_community_trial,
                    (configuration, disease, rand), num_trials)


def uniform_entry_point():
    """Run experiments on connected community networks with a unifrom degree distribution."""
    print('Running uniform connected community experiments')

    num_trials = 100
    disease = Disease(4, .2)
    rand = np.random.default_rng(0)
    configurations = [UniformConfiguration(rand, inner_bounds, outer_bounds, N_comm, num_comms)
                      for inner_bounds, outer_bounds, N_comm, num_comms
                      in [((10, 16), (2, 5), 25, 20),
                          ((10, 16), (5, 11), 25, 20),
                          ((5, 9), (2, 5), 10, 50),
                          ((5, 9), (5, 9), 10, 50)]]

    for configuration in configurations:
        print(configuration.name)
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
    def __init__(self, rand, inner_lam: Number, outer_lam: Number,
                 N_comm: int, num_communities: int) -> None:
        self._name = f'PoissonConfig(il={inner_lam}, ol={outer_lam} '\
                     f'N_comm={N_comm}, num_comms = {num_communities})'
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

        inner_bounds: The [lower, upper) bounds for the intracommunity degree distribution
        outer_bounds: The [lower, upper) bounds for the intercommunity degree distribution
        N_comm: The number of nodes in a community
        num_communities: The number of communities
        """
        self._name = f'UniformConfig(ib={inner_bounds}, ob={outer_bounds} '\
                     f'N_comm={N_comm}, num_comms={num_communities})'
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
        -> Optional[Tuple[float, float, float]]:
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
    net = networkgen.make_connected_community_network(inner_degrees, outer_degrees, rand)
    # If a network couldn't be successfully generated, return None to signal the failure
    if net is None:
        return None
    to_flicker = {(u, v) for u, v in net.edges if net.communities[u] != net.communities[v]}
    proportion_flickering = len(to_flicker) / net.E
    social_good = rate_social_good(net)

    behavior = StaticFlickerBehavior(net.M, to_flicker, (True, False), "Probs don't change this")
    avg_sus = np.mean([np.sum(simulate(net.M, make_starting_sir(net.N, 1),
                                       disease, behavior, sim_len, None, rand)[-1][0] > 0)
                       for _ in range(sims_per_trial)]) / net.N

    return proportion_flickering, avg_sus, social_good


def visual_inspection():
    N_comm = 25  # agents per community
    num_comms = 20  # number of communities
    rand = np.random.default_rng(0)
    inner_bounds = (10, 16)
    outer_bounds = (5, 11)
    configuration = UniformConfiguration(rand, inner_bounds, outer_bounds, N_comm, num_comms)

    for i in range(10):
        inner_degrees = configuration.make_inner_degrees()
        outer_degrees = configuration.make_outer_degrees()
        net = networkgen.make_connected_community_network(inner_degrees, outer_degrees, rand)
        if net is None:
            print('Failure')
            continue
        node_color = [COLORS[community] for community in net.communities.values()]
        visualize_network(G, None, name=str(i), block=False, node_color=node_color)  # type: ignore
    input('Done')


if __name__ == '__main__':
    try:
        uniform_entry_point()
        # poisson_entry_point()
        # visual_inspection()
    except KeyboardInterrupt:
        print('\nGood bye.')
    except EOFError:
        print('\nGood bye.')
