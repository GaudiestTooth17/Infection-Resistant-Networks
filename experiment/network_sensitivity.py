"""
The purpose of these experiments is to try to establish how much the outcome on a certain network
is to which agent starts infectious.
"""
import sys
from typing import Tuple
sys.path.append('')
import fileio as fio
import numpy as np
import matplotlib.pyplot as plt
from network import Network
import sim_dynamic as sd
from common import (MakeBarabasiAlbert, MakeConnectedCommunity, MakeRandomNetwork,
                    MakeWattsStrogatz, simulate_return_survival_rate)
import networkx as nx
from tqdm import tqdm
import itertools as it


def elitist_experiment():
    rng = np.random.default_rng()
    path = 'networks/elitist-500.txt'
    name = fio.get_network_name(path)
    G, _, communities = fio.read_network(path)
    net = Network(G, communities=communities)
    r, fp = 2, .75
    update_connections, uc_name = sd.PressureBehavior(net, r, fp), f'Pressure(r={r}, fp={fp})'
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
    r, fp = 2, .5
    disease = sd.Disease(4, .4)

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
    networks: Tuple[MakeRandomNetwork, ...] = (
        MakeConnectedCommunity(20, (15, 20), 25, (3, 6), rng),
        MakeConnectedCommunity(10, (5, 10), 50, (3, 6), rng),
        MakeWattsStrogatz(500, 4, .01),
        MakeWattsStrogatz(500, 4, .02),
        MakeWattsStrogatz(500, 5, .01),
        MakeBarabasiAlbert(500, 2),
        MakeBarabasiAlbert(500, 3),
        MakeBarabasiAlbert(500, 4)
    )

    print(f'Running {len(networks)*len(sir_strats)} experiments')
    for make_net, (strat_name, sir_strat) in it.product(networks, sir_strats):
        survival_rates = []
        for _ in tqdm(range(1000), desc=f'{make_net.class_name} & {strat_name}'):
            net = make_net()
            update_connections = sd.PressureBehavior(net, r, fp)
            sir0 = sir_strat(net)
            survival_rate = simulate_return_survival_rate(net, disease, update_connections,
                                                          rng, sir0)
            survival_rates.append(survival_rate)

        title = f'{strat_name} Patient 0\n{make_net.class_name} Survival Rates'
        plt.figure()
        plt.title(title)
        plt.xlim(0, 1.0)
        plt.ylim(0, 1000)
        plt.hist(survival_rates, bins=None)
        plt.savefig(title+'.png', format='png')


if __name__ == '__main__':
    # elitist_experiment()
    choose_infected_node()
