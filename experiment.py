"""
This file is for running experiments and reporting on them.
"""
from sim_dynamic import Disease, FlickerBehavior, make_starting_sir, simulate
from customtypes import Number
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Sequence
import os
import csv
from abc import ABC, abstractmethod
import networkgen
from tqdm import tqdm


@dataclass
class MassDiseaseTestingResult:
    network_class: str
    """A string describing what type of network the simulations were run on."""
    trial_to_results: Sequence[Sequence[int]]
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
            writer.writerows(self.trial_to_results)
            writer.writerow([])
            writer.writerow(self.trial_to_proportion_flickering_edges)


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


def main():
    N_comm = 10  # agents per community
    num_communities = 50  # number of communities
    num_trials = 1000
    sims_per_trial = 1000
    sim_len = 200
    rand = np.random.default_rng(420)
    configuration = PoissonConfiguration(f'Poisson: lam=10, {num_communities} comms, {N_comm} big',
                                         rand, 10, 10, num_communities, N_comm)
    disease = Disease(4, .2)

    trial_to_results: Sequence[Sequence[int]] = []
    trial_to_flickering_edges: Sequence[float] = []
    for _ in tqdm(range(num_trials)):
        inner_degrees = configuration.make_inner_degrees()
        outer_degrees = configuration.make_outer_degrees()
        G, communities = networkgen.make_connected_community_network(inner_degrees, outer_degrees, rand)
        to_flicker = {(u, v) for u, v in G.edges if communities[u] != communities[v]}
        trial_to_flickering_edges.append(len(to_flicker)/len(G.edges))
        M = nx.to_numpy_array(G)
        behavior = FlickerBehavior(M, to_flicker, (True, False),
                                   "Probs don't change this")
        trial_to_results.append([np.sum(simulate(M, make_starting_sir(len(M), 1),
                                                 disease, behavior,
                                                 sim_len, None)[-1][0])
                                 for _ in range(sims_per_trial)])
    experiment_results = MassDiseaseTestingResult(configuration.name, trial_to_results,
                                                  trial_to_flickering_edges)
    experiment_results.save('results')


if __name__ == '__main__':
    main()
