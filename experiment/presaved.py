import sys
sys.path.append('')
import os
from multiprocessing import Pool
from customtypes import ExperimentResults
from network import Network
from socialgood import rate_social_good
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import networkx as nx
from common import FlickerConfig, RandomFlickerConfig, StaticFlickerConfig
from sim_dynamic import Disease, simulate, make_starting_sir
import fileio as fio
from experiment.common import BasicExperimentResult
from tqdm import tqdm
import time
RESULTS_DIR = 'results'


def main():
    n_trials = 100
    max_steps = 100
    rng = np.random.default_rng(0)
    disease = Disease(4, .2)
    names = ('elitist-500', 'cavemen-50-10', 'spatial-network', 'cgg-500',
             'watts-strogatz-500-4-.1')
    paths = fio.network_names_to_paths(names)
    behavior_configs = (RandomFlickerConfig(.5, 'Random .5', rng),
                        StaticFlickerConfig((True, False), 'Static .5'))

    for net_name, path in zip(names, paths):
        G, _, communities = fio.read_network(path)
        net = Network(G)
        if communities is None:
            print(f'No community data for {net_name}. Skipping.', file=sys.stderr)
            continue
        to_flicker = tuple((u, v) for u, v in G.edges
                           if communities[u] != communities[v])
        proportion_flickering = len(to_flicker) / len(G.edges)
        social_good = rate_social_good(net)
        trial_to_pf = tuple(proportion_flickering for _ in range(n_trials))
        trial_to_sg = tuple(social_good for _ in range(n_trials))
        print(f'Running simulations for {net_name}.')
        for config in behavior_configs:
            behavior = config.make_behavior(net.M, to_flicker)
            sim_results = [get_final_stats(simulate(net.M, make_starting_sir(net.N, 1, rng),
                                                    disease, behavior, max_steps,
                                                    None, rng=rng))
                           for _ in tqdm(range(n_trials))]
            results = BasicExperimentResult(f'{net_name} {config.name}', sim_results,
                                            trial_to_pf, trial_to_sg)
            results.save_csv(RESULTS_DIR)
            results.save_box_plots(RESULTS_DIR)
            results.save_perc_sus_vs_social_good(RESULTS_DIR)


def get_final_stats(all_stats: Sequence[np.ndarray]) -> float:
    """Return the percentage of agents susceptible at the end of the simulation."""
    return np.sum(all_stats[-1][0] > 0) / len(all_stats[-1][0])


def entry_point():
    start_time = time.time()
    network_names = ('agent-generated-500',
                     'annealed-agent-generated-500',
                     'annealed-large-diameter',
                     'annealed-medium-diameter',
                     'annealed-short-diameter',
                     'cgg-500',
                     'watts-strogatz-500-4-.1',
                     'elitist-500',
                     'spatial-network',
                     'connected-comm-50-10',
                     'cavemen-50-10')
    network_paths = ['networks/'+name+'.txt' for name in network_names]
    # verify that all the networks exist
    found_errors = False
    for path in network_paths:
        if not os.path.isfile(path):
            print(f'{path} does not exist!')
            found_errors = True
    if found_errors:
        print('Fix errors before continuing')
        exit(1)

    flicker_configurations = [StaticFlickerConfig((True,), 'Static'),
                              StaticFlickerConfig((True, True, False), 'Two Thirds Flicker'),
                              StaticFlickerConfig((True, False), 'One Half Flicker'),
                              StaticFlickerConfig((True, False, False), 'One Third Flicker')]
    arguments = [(path, 1000, 500, Disease(4, .2), flicker_configurations,
                  flicker_configurations[0].name)
                 for path in network_paths]
    # use a maximum of 10 cores
    with Pool(min(len(arguments), 10)) as p:
        expirement_results = p.map(run_experiments, arguments)  # type: ignore

    results_dir = 'experiment results'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    for result in expirement_results:
        if result is not None:
            result.save(results_dir)

    print(f'Finished simulations ({time.time()-start_time}).')


def run_experiments(args: Tuple[str, int, int, Disease, Sequence[FlickerConfig], str])\
        -> Optional[ExperimentResults]:
    """
    Run a batch of experiments and return a tuple containing the network's name,
    number of flickering edges, and a mapping of behavior name to the final
    amount of susceptible nodes. Return None on failure.

    args: (path to the network,
           number of sims to run for each behavior,
           simulation length,
           disease,
           a sequence of configs for the flickers to use,
           the name of the baseline flicker to compare the other results to)
    """
    network_path, num_sims, sim_len, disease, flicker_configs, baseline_flicker_name = args
    G, layout, communities = fio.read_network(network_path)
    if layout is None:
        print(f'{fio.get_network_name(network_path)} has no layout.')
        return None
    if communities is None:
        print(f'{fio.get_network_name(network_path)} has no community data.')
        return None
    M = nx.to_numpy_array(G)
    intercommunity_edges = {(u, v) for u, v in G.edges if communities[u] != communities[v]}
    N = M.shape[0]

    behavior_to_results: Dict[str, Sequence[int]] = {}
    for config in flicker_configs:
        behavior = config.make_behavior(M, intercommunity_edges)
        # The tuple comprehension is pretty arcane, so here is an explanation.
        # Each entry is the sum of the number of entries in the final SIR where
        # the days in S are greater than 0. That is to say, the number of
        # susceptible agents at the end of the simulation.
        num_sus = tuple(np.sum(simulate(M,
                                        make_starting_sir(N, 1),
                                        disease,
                                        behavior,
                                        sim_len,
                                        None)[-1][0] > 0)
                        for _ in range(num_sims))
        behavior_to_results[behavior.name] = num_sus

    return ExperimentResults(fio.get_network_name(network_path), num_sims, sim_len,
                             len(intercommunity_edges)/len(G.edges), behavior_to_results,
                             baseline_flicker_name)


if __name__ == '__main__':
    main()
