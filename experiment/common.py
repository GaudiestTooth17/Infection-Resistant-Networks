import sys
sys.path.append('')
from dataclasses import dataclass
from network import Network
from sim_dynamic import (Disease, SimplePressureBehavior, RandomFlickerBehavior,
                         StaticFlickerBehavior, UpdateConnections, make_starting_sir, simulate)
from typing import (Any, Callable, Collection, List, Optional, Tuple, TypeVar,
                    Sequence, Dict)
from abc import ABC, abstractmethod
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.stats import wasserstein_distance
import os
import csv
from customtypes import Array, Number
from networkgen import make_connected_community_network
from pathlib import Path
import copy
import fileio as fio
T = TypeVar('T')


def _create_directory(directory: str):
    Path(directory).mkdir(parents=True, exist_ok=True)


class BasicExperimentResult:
    def __init__(self, name: str,
                 trial_to_perc_sus: Sequence[Number],
                 trial_to_proportion_flickering_edges: Sequence[Number],
                 trial_to_social_good: Sequence[Number]):
        """
        A class for aggregating and recording results of experiments on one type
        of network with a fixed flicker rate.

        network_class: A string describing what type of network the simulations were run on.
        trial_to_perc_sus: The percentage (or probably the average percentage)
                           of agents that ended the experiment susceptible.
        trial_to_proportion_flickering_edges: The trial should line up with the
                                              trial in trial_to_results.
        trial_to_social_good: The trial should line up with the trial in trial_to_results.
        """
        self.name = name
        self.trial_to_perc_sus = trial_to_perc_sus
        self.trial_to_proportion_flickering_edges = trial_to_proportion_flickering_edges
        self.trial_to_social_good = trial_to_social_good

    def save_csv(self, directory: str) -> None:
        """Save a CSV with the stored data."""
        _create_directory(directory)

        with open(os.path.join(directory, self.name+'.csv'), 'w', newline='') as file:
            writer = csv.writer(file, dialect=csv.excel)
            writer.writerow(['Percentage of Susceptible Agents'])
            writer.writerow(self.trial_to_perc_sus)
            writer.writerow(['Proportion Flickering Edges'])
            writer.writerow(self.trial_to_proportion_flickering_edges)
            writer.writerow(['Social Good'])
            writer.writerow(self.trial_to_social_good)

    def save_box_plots(self, directory: str) -> None:
        """Save box plots of all the stored data."""
        _create_directory(directory)

        plt.figure()
        plt.title(f'Percentage Suseptible for\n{self.name}')
        plt.boxplot(self.trial_to_perc_sus, notch=False)
        plt.savefig(os.path.join(directory,
                                 f'Percentage Suseptible for {self.name}.png'),
                    format='png')

        plt.figure()
        plt.title(f'Proportion Flickering Edges in\n{self.name}')
        plt.boxplot(self.trial_to_proportion_flickering_edges, notch=False)
        plt.savefig(os.path.join(directory,
                                 f'Proportion Flickering Edges in {self.name}.png'),
                    format='png')

        plt.figure()
        plt.title(f'Social Good on\n{self.name}')
        plt.boxplot(self.trial_to_social_good, notch=False)
        plt.savefig(os.path.join(directory,
                                 f'Social Good on {self.name}.png'),
                    format='png')

    def save_perc_sus_vs_social_good(self, directory: str,
                                     *, static_x: bool = True, static_y: bool = True) -> None:
        """Save a scatter plot of percentage susctible vs social good"""
        _create_directory(directory)

        plt.figure()
        plt.title(f'Resilience vs Social Good Trade-off Space\n{self.name}')
        plt.xlabel('Percentage Susceptible')
        plt.ylabel('Social Good')
        if static_x:
            plt.xlim(0, 1)
        if static_y:
            plt.ylim(0, 1)
        plt.scatter(self.trial_to_perc_sus, self.trial_to_social_good)
        plt.savefig(os.path.join(directory,
                                 f'R vs SG Trade off Space for {self.name}.png'),
                    format='png')


class FlickerComparisonResult:
    def __init__(self, network_name: str,
                 sims_per_behavior: int,
                 sim_len: int,
                 proportion_flickering_edges: float,
                 behavior_to_survival_rate: Dict[str, Sequence[float]],
                 baseline_behavior: str):
        """
        A class for gathering data on the effectives of different behaviors in comparision
        to each other.

        network_name
        sims_per_behavior
        behavior_to_num_sus: How many agents were still susceptible at the end of
                            each simulation with the specified behavior.
        baseline_behavior: The name of the behavior to computer the
                        Wasserstein distance of the others against.
        """
        self.network_name = network_name
        self.sims_per_behavior = sims_per_behavior
        self.sim_len = sim_len
        self.proportion_flickering_edges = proportion_flickering_edges
        self.behavior_to_survival_rate = behavior_to_survival_rate
        # Fail early if an incorrect name is supplied.
        if baseline_behavior not in behavior_to_survival_rate:
            print(f'{baseline_behavior} is not in {list(behavior_to_survival_rate.keys())}.'
                  'Fix this before continuing.')
            exit(1)
        self.baseline_behavior = baseline_behavior

    def save(self, directory: str, with_histograms: bool = False) -> None:
        """Save a histogram and a text file with analysis information in directory."""
        path = os.path.join(directory, self.network_name)
        _create_directory(path)

        # File Heading
        file_lines = [f'Name: {self.network_name}\n',
                      f'Number of sims per behavior: {self.sims_per_behavior}\n',
                      f'Simulation length: {self.sim_len}\n'
                      f'Proportion of edges flickering: {self.proportion_flickering_edges:.4f}\n\n']
        baseline_distribution = self.behavior_to_survival_rate[self.baseline_behavior]
        for behavior_name, results in self.behavior_to_survival_rate.items():
            # possibly save histograms
            if with_histograms:
                plt.figure()
                title = f'{self.network_name} {behavior_name}\n'\
                    f'sims={self.sims_per_behavior} sim_len={self.sim_len}'
                plt.title(title)
                plt.xlabel('Number of Susceptible Agents')
                plt.ylabel('Frequency')
                plt.hist(results, bins=None)
                plt.savefig(os.path.join(path, title+'.png'), format='png')

            # create a text entry for each behavior
            file_lines += [f'{behavior_name}\n',
                           f'Min:{np.min(results) : >20}\n',
                           f'Max:{np.max(results) : >20}\n',
                           f'Median:{np.median(results) : >20}\n',
                           f'Mean:{np.mean(results) : >20.3f}\n',
                           f'EMD from {self.baseline_behavior}:'
                           f'{wasserstein_distance(results, baseline_distribution) : >20.3f}\n\n']

        # save text entries
        with open(os.path.join(path, f'Report on {self.network_name}.txt'), 'w') as file:
            file.writelines(file_lines)


class PressureComparisonResult:
    def __init__(self, network_name: str,
                 disease: Disease,
                 sims_per_behavior: int,
                 behavior_to_survival_rate: Dict[str, Array],
                 baseline_behavior: str):
        """
        A class for gathering data on the effectives of different behaviors in comparision
        to each other.

        network_name
        sims_per_behavior
        behavior_to_num_sus: How many agents were still susceptible at the end of
                            each simulation with the specified behavior.
        baseline_behavior: The name of the behavior to computer the
                        Wasserstein distance of the others against.
        """
        self.network_name = network_name
        self.sims_per_behavior = sims_per_behavior
        self.behavior_to_survival_rate = behavior_to_survival_rate
        # Fail early if an incorrect name is supplied.
        if baseline_behavior not in behavior_to_survival_rate:
            print(f'{baseline_behavior} is not in {list(behavior_to_survival_rate.keys())}.'
                  'Fix this before continuing.')
            exit(1)
        self.baseline_behavior = baseline_behavior
        self.disease = disease

    def save(self, directory: str, with_histograms: bool = False) -> None:
        """Save a histogram and a text file with analysis information in directory."""
        path = os.path.join(directory, self.network_name)
        _create_directory(path)

        # File Heading
        file_lines = [f'Name: {self.network_name}\n',
                      f'Disease: {self.disease}\n',
                      f'Number of sims per behavior: {self.sims_per_behavior}\n\n']
        baseline_distribution = self.behavior_to_survival_rate[self.baseline_behavior]
        for behavior_name, results in self.behavior_to_survival_rate.items():
            # possibly save histograms
            if with_histograms:
                plt.figure()
                title = f'{self.network_name}\n{behavior_name} sims={self.sims_per_behavior}'
                plt.title(title)
                plt.xlabel('Survival Rate')
                plt.ylabel('Frequency')
                plt.hist(results, bins=None)
                plt.savefig(os.path.join(path, title+'.png'), format='png')

            # create a text entry for each behavior
            file_lines += [f'{behavior_name}\n',
                           f'{"Min:":<20}{np.min(results):.3f}\n',
                           f'{"Max:":<20}{np.max(results):.3f}\n',
                           f'{"Median:":<20}{np.median(results):.3f}\n',
                           f'{"Mean:":<20}{np.mean(results):.3f}\n',
                           f'{f"EMD from {self.baseline_behavior}:":<20}'
                           f'{wasserstein_distance(results, baseline_distribution):.3f}\n\n']

        # save text entries
        with open(os.path.join(path, f'Report on {self.network_name}.txt'), 'w') as file:
            file.writelines(file_lines)

    def save_raw(self, directory: str) -> None:
        path = os.path.join(directory, self.network_name)
        _create_directory(path)

        with open(os.path.join(path, 'raw_data.csv'), 'w', newline='') as file:
            writer = csv.writer(file, dialect=csv.excel)
            for behavior, survival_rates in self.behavior_to_survival_rate.items():
                writer.writerow([behavior])
                writer.writerow(survival_rates)


def safe_run_trials(name: str, trial_func: Callable[[T], Optional[Tuple[float, float, float]]],
                    args: T, num_trials: int, max_failures: int = 10) -> None:
    """Run trials until too many failures occur, exit if this happens."""
    results: List[Tuple[float, float, float]] = []
    failures_since_last_success = 0
    pbar = tqdm(total=num_trials, desc=f'Failures: {failures_since_last_success}')
    while len(results) < num_trials:
        if failures_since_last_success >= max_failures:
            print(f'Failure limit has been reached. {name} is not feasible.')
            exit(1)

        result = trial_func(args)
        if result is None:
            failures_since_last_success += 1
            update_amount = 0
        else:
            results.append(result)
            failures_since_last_success = 0
            update_amount = 1
        pbar.set_description(f'Failures: {failures_since_last_success}')
        pbar.update(update_amount)

    trial_to_flickering_edges, trial_to_avg_sus, trial_to_social_good = zip(*results)
    experiment_results = BasicExperimentResult(name, trial_to_avg_sus,
                                               trial_to_flickering_edges, trial_to_social_good)
    experiment_results.save_perc_sus_vs_social_good('results')


def simulate_return_survival_rate(net: Network, disease: Disease,
                                  behavior: UpdateConnections, rng,
                                  sir0: Optional[np.ndarray] = None) -> float:
    if sir0 is None:
        sir0 = make_starting_sir(net.N, 1, rng)
    return np.sum(simulate(net.M, sir0, disease, behavior, 100, None, rng)[-1][0] > 0) / net.N


class FlickerConfig(ABC):
    @abstractmethod
    def make_behavior(self, M: np.ndarray,
                      edges_to_flicker: Collection[Tuple[int, int]])\
            -> UpdateConnections:
        """Return a some flicker behavior."""
        pass


class StaticFlickerConfig(FlickerConfig):
    def __init__(self, flicker_pattern: Sequence[bool],
                 name: Optional[str] = None) -> None:
        self.flicker_pattern = flicker_pattern
        self.name = name

    def make_behavior(self, M: np.ndarray,
                      edges_to_flicker: Collection[Tuple[int, int]])\
            -> StaticFlickerBehavior:
        return StaticFlickerBehavior(M, edges_to_flicker,
                                     self.flicker_pattern,
                                     self.name)


class RandomFlickerConfig(FlickerConfig):
    def __init__(self, flicker_probability: float,
                 name: Optional[str] = None,
                 rand: Optional[Any] = None):
        self.flicker_probability = flicker_probability
        self.name = name
        self.rand = rand

    def make_behavior(self, M: np.ndarray,
                      edges_to_flicker: Collection[Tuple[int, int]])\
            -> RandomFlickerBehavior:
        if self.rand is None:
            return RandomFlickerBehavior(M, edges_to_flicker,
                                         self.flicker_probability,
                                         self.name)
        return RandomFlickerBehavior(M, edges_to_flicker,
                                     self.flicker_probability,
                                     self.name, self.rand)


@dataclass
class PressureConfig:
    radius: int
    flicker_probability: float
    rng: Any
    name: Optional[str] = None

    def make_behavior(self, net: Network) -> SimplePressureBehavior:
        return SimplePressureBehavior(net, self.radius, self.flicker_probability,
                                      self.rng, self.name)


class MakeNetwork(ABC):
    """
    Interface for classes that create networks and keep track of the type of network they create.
    """
    @property
    @abstractmethod
    def class_name(self) -> str:
        """A description of the class of random network, or just a name for a non random network."""
        pass

    @property
    @abstractmethod
    def seed(self) -> Optional[int]:
        """
        The seed used to make the random network. This will be the seed for
        default_rng if it is a custom random network, or the seed passed to the
        NetworkX function. None is returned for networks that aren't generated
        randomly.
        """
        return None

    @abstractmethod
    def __call__(self) -> Network:
        """Return the appropriate network."""
        pass


class MakeConnectedCommunity(MakeNetwork):
    def __init__(self, community_size: int, inner_bounds: Tuple[int, int],
                 num_comms: int, outer_bounds: Tuple[int, int], seed: int):
        self._community_size = community_size
        self._inner_bounds = inner_bounds
        self._num_comms = num_comms
        self._outer_bounds = outer_bounds
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._class_name = f'ConnComm(N_comm={community_size},ib={inner_bounds},'\
                           f'num_comms={num_comms},ob={outer_bounds})'

    @property
    def class_name(self) -> str:
        return self._class_name

    @property
    def seed(self) -> int:
        return self._seed

    def __call__(self) -> Network:
        id_dist = self._rng.integers(self._inner_bounds[0], self._inner_bounds[1],
                                     self._community_size, endpoint=True)
        if np.sum(id_dist) % 2 > 0:
            id_dist[np.argmin(id_dist)] += 1
        od_dist = self._rng.integers(self._outer_bounds[0], self._outer_bounds[1],
                                     self._num_comms, endpoint=True)
        if np.sum(od_dist) % 2 > 0:
            od_dist[np.argmin(od_dist)] += 1

        net = make_connected_community_network(id_dist, od_dist, self._rng)
        if net is None:
            raise Exception('This should not have happened.')
        return net


class MakeBarabasiAlbert(MakeNetwork):
    def __init__(self, N: int, m: int, seed: int):
        self._N = N
        self._m = m
        self._seed = seed
        self._class_name = f'BarabasiAlbert(N={N},m={m})'

    @property
    def class_name(self) -> str:
        return self._class_name

    @property
    def seed(self) -> int:
        return self._seed

    def __call__(self) -> Network:
        return Network(nx.barabasi_albert_graph(self._N, self._m, self._seed))


class MakeWattsStrogatz(MakeNetwork):
    def __init__(self, N: int, k: int, p: float, seed: int):
        self._N = N
        self._k = k
        self._p = p
        self._seed = seed
        self._class_name = f'WattsStrogatz(N={N},k={k},p={p})'

    @property
    def class_name(self) -> str:
        return self._class_name

    @property
    def seed(self) -> int:
        return self._seed

    def __call__(self) -> Network:
        return Network(nx.watts_strogatz_graph(self._N, self._k, self._p, self._seed))


class MakeErdosRenyi(MakeNetwork):
    def __init__(self, N: int, p: float, seed: int) -> None:
        self._N = N
        self._p = p
        self._seed = seed
        self._class_name = f'ErdosRenyi(N={N},p={p})'

    @property
    def class_name(self) -> str:
        return self._class_name

    @property
    def seed(self) -> int:
        return self._seed

    def __call__(self) -> Network:
        return Network(nx.erdos_renyi_graph(self._N, self._p, self._seed))


class MakeGrid(MakeNetwork):
    def __init__(self, m: int, n: int) -> None:
        self._m = m
        self._n = n
        self._class_name = f'Grid(n={n},m={m})'
        # There is no randomness, so just save the network once it is generated once.
        self._net: Optional[Network] = None

    @property
    def class_name(self) -> str:
        return self._class_name

    @property
    def seed(self) -> None:
        return None

    def __call__(self) -> Network:
        if self._net is None:
            self._net = Network(nx.grid_2d_graph(self._m, self._n))
        # Make a copy so that if someone really wants to mutate the Network, it
        # won't screw up future return values.
        return copy.deepcopy(self._net)


class LoadNetwork(MakeNetwork):
    def __init__(self, name: str):
        self._name = name
        self._net = None

    @property
    def class_name(self) -> str:
        return self._name

    @property
    def seed(self) -> None:
        return None

    def __call__(self) -> Network:
        if self._net is None:
            path = fio.network_names_to_paths((self._name,))[0]
            self._net = fio.read_network(path)
        return self._net
