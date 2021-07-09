"""
This file is for running experiments and reporting on them.
"""
from tqdm.std import tqdm
from sim_dynamic import Disease, FlickerBehavior, make_starting_sir, simulate
from customtypes import Agent, Number
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, TypeVar
import os
import csv
from abc import ABC, abstractmethod
import networkgen
import partitioning
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
import agentbasedgen as abg
T = TypeVar('T')


@dataclass
class MassDiseaseTestingResult:
    network_class: str
    """A string describing what type of network the simulations were run on."""
    trial_to_results: Sequence[Number]
    """Each result is actually a collection of results from one instance of the  network class."""
    trial_to_proportion_flickering_edges: Sequence[float]
    """The trial should line up with the trial in trial_to_results."""

    def save(self, directory: str) -> None:
        """
        Save a CSV with trials_to_results stored in rows, a blank row, and
        trial_to_proportion_flickering_edges stored next in a single row.
        """
        if not os.path.exists(directory):
            os.mkdir(directory)

        with open(self.network_class+'.csv', 'w', newline='') as file:
            writer = csv.writer(file, dialect=csv.excel)
            writer.writerow(['Results'])
            writer.writerow(self.trial_to_results)
            writer.writerow(['Proportion Flickering Edges'])
            writer.writerow(self.trial_to_proportion_flickering_edges)

        plt.figure()
        plt.title(f'Results for {self.network_class}')
        plt.boxplot(self.trial_to_results, notch=False)
        plt.savefig(f'Results for {self.network_class}.png', format='png')

        plt.figure()
        plt.title(f'Proportion Flickering Edges for {self.network_class}')
        plt.boxplot(self.trial_to_proportion_flickering_edges, notch=False)
        plt.savefig(f'Proportion Flickering Edges for {self.network_class}', format='png')


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


def run_agent_generated_trial(args: Tuple[Disease, abg.Behavior, int, Any]) -> Tuple[float, float]:
    """
    args: (disease to use in the simulation,
           the behavior agents have when generating the network,
           the number of agents in the network,
           an instance of np.random.default_rng)
    """
    disease, agent_behavior, N, rand = args
    sim_len = 200
    sims_per_trial = 150
    G = None
    while G is None:
        G = abg.make_agent_generated_network(N, agent_behavior)

    to_flicker = partitioning.fluidc_partition(G, 50)
    proportion_flickering = len(to_flicker) / len(G.edges)
    M = nx.to_numpy_array(G)

    network_behavior = FlickerBehavior(M, to_flicker, (True, False), "Probs don't change this")
    avg_sus = np.mean([np.sum(simulate(M, make_starting_sir(len(M), 1),
                                       disease, network_behavior, sim_len, None, rand)[-1][0] > 0)
                       for _ in range(sims_per_trial)])

    return proportion_flickering, avg_sus


def run_social_circles_trial(args: Tuple[Dict[Agent, int],
                                         Tuple[int, int],
                                         Disease,
                                         Any]) -> Tuple[float, float]:
    agent_to_quantity, grid_dims, disease, rand = args
    sim_len = 200
    sims_per_trial = 150

    sc_results = None
    while sc_results is None:
        sc_results = networkgen.make_social_circles_network(agent_to_quantity, grid_dims, rand=rand)
    G, _, _ = sc_results

    to_flicker = partitioning.fluidc_partition(G, 25)
    proportion_flickering = len(to_flicker) / len(G.edges)
    M = nx.to_numpy_array(G)

    network_behavior = FlickerBehavior(M, to_flicker, (True, False), "Probs don't change this")
    avg_sus = np.mean([np.sum(simulate(M, make_starting_sir(len(M), 1),
                                       disease, network_behavior, sim_len, None, rand)[-1][0] > 0)
                       for _ in range(sims_per_trial)])

    return proportion_flickering, avg_sus


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
    inner_bounds = (1, 9)
    outer_bounds = (1, 4)
    configuration = UniformConfiguration(rand, inner_bounds, outer_bounds, N_comm, num_comms)
    disease = Disease(4, .2)

    safe_run_trials(configuration.name, run_connected_community_trial,
                    (configuration, disease, rand), num_trials)


def agent_generated_entry_point():
    print('Running agent generated experiments')
    start_time = time.time()
    num_trials = 1000
    rand = np.random.default_rng(1337)
    N = 500
    lb_connection = 4
    ub_connection = 6
    steps_to_stability = 20
    agent_behavior = abg.TimeBasedBehavior(N, lb_connection, ub_connection,
                                           steps_to_stability, rand)
    disease = Disease(4, .2)

    safe_run_trials(f'Agentbased {N}-{lb_connection}-{ub_connection}-{steps_to_stability}',
                    run_agent_generated_trial, (disease, agent_behavior, N, rand), num_trials)

    print(f'Finished experiments with agent generated networks ({time.time()-start_time} s).')


def social_circles_entry_point():
    print('Running social circles experiments.')
    num_trials = 1000
    rand = np.random.default_rng(0xdeadbeef)
    N = 500
    N_purple = int(N * .1)
    N_blue = int(N * .2)
    N_green = N - N_purple - N_blue
    agents = {Agent('green', 30): N_green,
              Agent('blue', 40): N_blue,
              Agent('purple', 50): N_purple}
    grid_dim = (int(N/.003), int(N/.003))  # the denominator is the desired density
    disease = Disease(4, .2)

    safe_run_trials(f'Social Circles (Elitist) {grid_dim} {N}', run_social_circles_trial,
                    (agents, grid_dim, disease, rand), num_trials)


def safe_run_trials(name: str, trial_func: Callable[[T], Optional[Tuple[float, float]]],
                    args: T, num_trials: int) -> None:
    """Run trials until too many failures occur, exit if this happens."""
    results = []
    failures_since_last_success = 0
    pbar = tqdm(total=num_trials)
    while len(results) < num_trials:
        if failures_since_last_success > 100:
            print(f'Failure limit has been reached. {name} is not feasible.')
            exit(1)

        result = trial_func(args)
        if result is None:
            failures_since_last_success += 1
        else:
            results.append(result)
            pbar.update()

    trial_to_flickering_edges, trial_to_avg_sus = zip(*results)
    experiment_results = MassDiseaseTestingResult(name, trial_to_avg_sus,
                                                  trial_to_flickering_edges)
    experiment_results.save('results')


def main():
    # poisson_entry_point()
    uniform_entry_point()
    # agent_generated_entry_point()
    # social_circles_entry_point()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nGood bye.')
    except EOFError:
        print('\nGood bye.')
