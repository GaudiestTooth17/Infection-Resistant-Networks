from sim_dynamic import RandomFlickerBehavior, StaticFlickerBehavior, UpdateConnections
from typing import (Any, Callable, Collection, List, Optional, Tuple, TypeVar,
                    Sequence)
from abc import ABC, abstractmethod
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import sys
sys.path.append('')
from customtypes import Number
T = TypeVar('T')


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
        self._create_directory(directory)

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
        self._create_directory(directory)

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
        self._create_directory(directory)

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

    @staticmethod
    def _create_directory(directory):
        if not os.path.exists(directory):
            os.mkdir(directory)


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
