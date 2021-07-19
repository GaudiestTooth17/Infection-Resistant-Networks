import sys
sys.path.append('')
from socialgood import rate_social_good
from typing import Sequence
import numpy as np
import networkx as nx
from common import RandomFlickerConfig, StaticFlickerConfig
from sim_dynamic import Disease, simulate, make_starting_sir
import fileio as fio
from experiment.common import ExperimentResult
from tqdm import tqdm
RESULTS_DIR = 'results'


def main():
    n_trials = 100
    max_steps = 100
    rng = np.random.default_rng(0)
    disease = Disease(4, .2)
    names = ('elitist-500', 'student-interaction-friday', 'cavemen-50-10',
             'spatial-network', 'cgg-500', 'watts-strogatz-500-4-.1')
    paths = fio.network_names_to_paths(names)
    behavior_configs = (RandomFlickerConfig(.5, 'Rand .5', rng),
                        StaticFlickerConfig((True, False), 'Static .5'))

    for net_name, path in zip(names, paths):
        G, _, communities = fio.read_network(path)
        M = nx.to_numpy_array(G)
        if communities is None:
            print(f'No community data for {net_name}. Skipping.', file=sys.stderr)
            continue
        to_flicker = tuple((u, v) for u, v in G.edges
                           if communities[u] != communities[v])
        proportion_flickering = len(to_flicker) / len(G.edges)
        social_good = rate_social_good(G)
        trial_to_pf = tuple(proportion_flickering for _ in range(n_trials))
        trial_to_sg = tuple(social_good for _ in range(n_trials))
        print(f'Running simulations for {net_name}.')
        for config in behavior_configs:
            behavior = config.make_behavior(M, to_flicker)
            sim_results = [get_final_stats(simulate(M, make_starting_sir(len(M), 1, rng),
                                                    disease, behavior, max_steps,
                                                    None, rand=rng))
                           for _ in tqdm(range(n_trials))]
            results = ExperimentResult(f'{net_name} {config.name}', sim_results,
                                       trial_to_pf, trial_to_sg)
            results.save_csv(RESULTS_DIR)
            results.save_box_plots(RESULTS_DIR)
            results.save_perc_sus_vs_social_good(RESULTS_DIR)


def get_final_stats(all_stats: Sequence[np.ndarray]) -> float:
    """Return the percentage of agents susceptible at the end of the simulation."""
    return np.sum(all_stats[-1][0] > 0) / len(all_stats[-1][0])


if __name__ == '__main__':
    main()
