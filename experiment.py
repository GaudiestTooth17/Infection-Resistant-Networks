"""
This file is for running experiments and reporting on them.
"""
from sim_dynamic import Disease, FlickerBehavior, make_starting_sir, simulate
from customtypes import ExperimentResults, Number
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Any, Sequence, Tuple
import os
import csv
from abc import ABC, abstractmethod
import networkgen
import partitioning
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
import agentbasedgen as abg


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


def run_poisson_trial(args: Tuple[PoissonConfiguration, Disease, Any]) -> Tuple[float, float]:
    """
    args: (PoissonConfiguration to use,
           disease to use)
    """
    sim_len = 200
    sims_per_trial = 250
    configuration, disease, rand = args

    inner_degrees = configuration.make_inner_degrees()
    outer_degrees = configuration.make_outer_degrees()
    G, communities = networkgen.make_connected_community_network(inner_degrees, outer_degrees,
                                                                 rand)
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


def poisson_entry_point():
    start_time = time.time()
    N_comm = 10  # agents per community
    num_comms = 50  # number of communities
    num_trials = 1000
    rand = np.random.default_rng(420)
    lam = 5  # this is lambda for both the inner and outer degree distributions
    configuration = PoissonConfiguration(f'Poisson: lam={lam}, {num_comms} comms, {N_comm} big',
                                         rand, lam, lam, num_comms, N_comm)
    disease = Disease(4, .2)

    with Pool(3) as p:
        results = p.map(run_poisson_trial, [(configuration, disease, rand)
                                            for _ in range(num_trials)],
                        num_trials//3)
    trial_to_flickering_edges, trial_to_avg_sus = zip(*results)
    experiment_results = MassDiseaseTestingResult(configuration.name, trial_to_avg_sus,
                                                  trial_to_flickering_edges)
    experiment_results.save('results')
    print(f'Finished ({time.time()-start_time} s).')


def agent_generated_entry_point():
    start_time = time.time()
    num_trials = 1000
    rand = np.random.default_rng(1337)
    N = 500
    lb_connection = 4
    ub_connection = 6
    steps_to_stability = 10
    agent_behavior = abg.TimeBasedBehavior(N, lb_connection, ub_connection,
                                           steps_to_stability, rand)
    disease = Disease(4, .2)

    with Pool(5) as p:
        results = p.map(run_agent_generated_trial, [(disease, agent_behavior, N, rand)
                                                    for _ in range(num_trials)],
                        num_trials//5)
    trial_to_flickering_edges, trial_to_avg_sus = zip(*results)
    experiment_results = MassDiseaseTestingResult(f'Agentbased {N}-{lb_connection}-{ub_connection}'
                                                  f'-{steps_to_stability}',
                                                  trial_to_avg_sus, trial_to_flickering_edges)
    experiment_results.save('results')
    print(f'Finished ({time.time()-start_time} s).')


def main():
    # poisson_entry_point()
    agent_generated_entry_point()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nGood bye.')
    except EOFError:
        print('\nGood bye.')
