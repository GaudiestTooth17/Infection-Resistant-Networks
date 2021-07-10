from typing import Callable, Optional, Tuple, TypeVar, Sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from customtypes import Number
import os
import csv
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


def safe_run_trials(name: str, trial_func: Callable[[T], Optional[Tuple[float, float]]],
                    args: T, num_trials: int, max_failures: int = 10) -> None:
    """Run trials until too many failures occur, exit if this happens."""
    results = []
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

    trial_to_flickering_edges, trial_to_avg_sus = zip(*results)
    experiment_results = MassDiseaseTestingResult(name, trial_to_avg_sus,
                                                  trial_to_flickering_edges)
    experiment_results.save('results')
