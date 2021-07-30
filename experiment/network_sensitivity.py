"""
The purpose of these experiments is to try to establish how much the outcome on a certain network
is to which agent starts infectious.
"""
import sys
sys.path.append('')
from typing import Tuple
import fileio as fio
import numpy as np
import matplotlib.pyplot as plt
from network import Network
import sim_dynamic as sd
from common import (MakeBarabasiAlbert, MakeConnectedCommunity, MakeErdosRenyi, MakeGrid, MakeNetwork,
                    MakeWattsStrogatz, simulate_return_survival_rate)
import networkx as nx
from tqdm import tqdm
import itertools as it
import os
import csv


def elitist_experiment():
    rng = np.random.default_rng()
    path = 'networks/elitist-500.txt'
    name = fio.get_network_name(path)
    net = fio.read_network(path)
    r, fp = 2, .75
    update_connections, uc_name = sd.SimplePressureBehavior(net, r, fp), f'Pressure(r={r}, fp={fp})'
    # update_connections, uc_name = sd.no_update, 'Static'
    disease = sd.Disease(4, .2)
    sir0 = sd.make_starting_sir(net.N, (0,), rng)
    survival_rates = np.array([simulate_return_survival_rate(net, disease, update_connections,
                                                             rng, sir0)
                               for _ in range(500)])
    title = f'{disease} {uc_name}\n{name} Survival Rates'
    plt.title(title)
    plt.boxplot(survival_rates)
    plt.savefig(title+'.png', format='png')


def choose_infected_node():
    """This experiment is for choosing nodes with specific attributes to be patient 0"""
    rng = np.random.default_rng()
    r, fp = 2, .25
    disease = sd.Disease(4, .3)
    n_trials = 500

    def choose_by_centrality(net, centrality_measure, max_or_min):
        degrees = dict(centrality_measure(net.G))  # type: ignore
        patient0 = max_or_min(degrees.keys(), key=lambda k: degrees[k])
        return sd.make_starting_sir(net.N, (patient0,), rng)

    sir_strats = (
        ('Highest Degree', lambda net: choose_by_centrality(net, nx.degree_centrality, max)),
        ('Random', lambda net: sd.make_starting_sir(net.N, 1, rng)),
        ('Lowest Degree', lambda net: choose_by_centrality(net, nx.degree_centrality, min)),
        ('Highest Eigenvector Centrality',
         lambda net: choose_by_centrality(net, nx.eigenvector_centrality_numpy, max)),
        ('Lowest Eigenvector Centrality',
         lambda net: choose_by_centrality(net, nx.eigenvector_centrality_numpy, min))
    )
    networks: Tuple[MakeNetwork, ...] = (
        MakeConnectedCommunity(20, (15, 20), 25, (3, 6), rng),
        MakeConnectedCommunity(10, (5, 10), 50, (3, 6), rng),
        MakeWattsStrogatz(500, 4, .01),
        MakeWattsStrogatz(500, 4, .02),
        MakeWattsStrogatz(500, 5, .01),
        MakeBarabasiAlbert(500, 2),
        MakeBarabasiAlbert(500, 3),
        MakeBarabasiAlbert(500, 4),
        MakeErdosRenyi(500, .01),
        MakeErdosRenyi(500, .02),
        MakeErdosRenyi(500, .03),
        MakeGrid(25, 20),
        MakeGrid(50, 10),
        MakeGrid(100, 5)
    )[-6:]

    print(f'Running {len(networks)*len(sir_strats)} experiments')
    experiment_to_survival_rates = {}
    for make_net, (strat_name, sir_strat) in it.product(networks, sir_strats):
        survival_rates = []
        for _ in tqdm(range(n_trials), desc=f'{make_net.class_name} & {strat_name}'):
            net = make_net()
            update_connections = sd.SimplePressureBehavior(net, r, fp)
            sir0 = sir_strat(net)
            survival_rate = simulate_return_survival_rate(net, disease, update_connections,
                                                          rng, sir0)
            survival_rates.append(survival_rate)

        title = f'{strat_name} Patient 0\n{make_net.class_name} Survival Rates'
        plt.figure()
        plt.title(title)
        plt.xlim(0, 1.0)
        plt.ylim(0, n_trials)
        plt.hist(survival_rates, bins=None)
        plt.savefig(os.path.join('results', title+'.png'), format='png')
        experiment_to_survival_rates[title] = survival_rates

    # save the raw data
    with open(os.path.join('results', 'patient-0-sensitivity.csv'), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for experiment_name, survival_rates in experiment_to_survival_rates.items():
            writer.writerow([experiment_name])
            writer.writerow(survival_rates)


if __name__ == '__main__':
    # elitist_experiment()
    choose_infected_node()
