"""
The purpose of these experiments is to try to establish how much the outcome on a certain network
is to which agent starts infectious.
"""
import sys
sys.path.append('')
from typing import List, Tuple, Union
import fileio as fio
import numpy as np
import matplotlib.pyplot as plt
import sim_dynamic as sd
from common import (LoadNetwork, MakeBarabasiAlbert, MakeConnectedCommunity,
                    MakeErdosRenyi, MakeGrid, MakeNetwork, MakeWattsStrogatz, SimplePressureConfig,
                    calc_entropy, run_sim_batch, simulate_return_survival_rate)
import networkx as nx
from tqdm import tqdm
import itertools as it
import os
import csv
import tarfile
import time


def elitist_experiment():
    rng = np.random.default_rng()
    path = 'networks/elitist-500.txt'
    name = fio.get_network_name(path)
    net = fio.read_network(path)
    r, fp = 2, .75
    update_connections, uc_name = (sd.SimplePressureBehavior(net, rng, r, fp),
                                   f'Pressure(r={r}, fp={fp})')
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
    seed = 0

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
        MakeWattsStrogatz(500, 4, .01, seed),
        MakeWattsStrogatz(500, 4, .02, seed),
        MakeWattsStrogatz(500, 5, .01, seed),
        MakeBarabasiAlbert(500, 2, seed),
        MakeBarabasiAlbert(500, 3, seed),
        MakeBarabasiAlbert(500, 4, seed),
        MakeErdosRenyi(500, .01, seed),
        MakeErdosRenyi(500, .02, seed),
        MakeErdosRenyi(500, .03, seed),
        MakeGrid(25, 20),
        MakeGrid(50, 10),
        MakeGrid(100, 5),
        LoadNetwork('evolved-low-communicability-100')
    )[-1:]

    print(f'Running {len(networks)*len(sir_strats)} experiments')
    experiment_to_survival_rates = {}
    for make_net, (strat_name, sir_strat) in it.product(networks, sir_strats):
        survival_rates = []
        for _ in tqdm(range(n_trials), desc=f'{make_net.class_name} & {strat_name}'):
            net = make_net()
            update_connections = sd.SimplePressureBehavior(net, rng, r, fp)
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


def centrality_plots():
    seed = 1
    centrality_name, calc_cntrlty = 'Eigenvector Centrality', nx.eigenvector_centrality_numpy
    # centrality_name, calc_cntrlty = 'Degree Centrality', lambda G: dict(nx.degree_centrality(G))
    make_network_funcs = (
        MakeConnectedCommunity(20, (15, 20), 25, (3, 6), seed),
        MakeConnectedCommunity(10, (5, 10), 50, (3, 6), seed),
        MakeWattsStrogatz(500, 4, .01, seed),
        MakeWattsStrogatz(500, 4, .02, seed),
        MakeWattsStrogatz(500, 5, .01, seed),
        MakeBarabasiAlbert(500, 2, seed),
        MakeBarabasiAlbert(500, 3, seed),
        MakeBarabasiAlbert(500, 4, seed),
        MakeErdosRenyi(500, .01, seed),
        MakeErdosRenyi(500, .02, seed),
        MakeErdosRenyi(500, .03, seed),
        MakeGrid(25, 20),
        MakeGrid(50, 10),
        MakeGrid(100, 5)
    )
    n_top_nodes = 10
    class_to_centrality_dist = {}
    for make_network in tqdm(make_network_funcs, desc=centrality_name):
        centralities = list(it.chain(*(sorted(calc_cntrlty(make_network().G).values(),
                                              reverse=True)[:n_top_nodes]
                                       for _ in range(1000))))
        class_to_centrality_dist[make_network.class_name] = centralities

    with open(f'{centrality_name}.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for class_name, centrality_dist in class_to_centrality_dist.items():
            title = f'{class_name}\n{centrality_name} of the Highest {n_top_nodes} nodes'
            plt.figure()
            plt.ylim(0, 1.0)
            plt.title(title)
            plt.ylabel(centrality_name)
            plt.boxplot(centrality_dist)
            plt.savefig(title+'.png', format='png')

            writer.writerow([class_name])
            writer.writerow(centrality_dist)


def infection_entropy_vs_communicability():
    classes = (
        'BarabasiAlbert(N=500,m=2)',
        'BarabasiAlbert(N=500,m=3)',
        'BarabasiAlbert(N=500,m=4)',
        'ConnComm(N_comm=10,ib=(5, 10),num_comms=50,ob=(3, 6))',
        'ConnComm(N_comm=20,ib=(15, 20),num_comms=25,ob=(3, 6))',
        'ErdosRenyi(N=500,p=0.01)',
        'ErdosRenyi(N=500,p=0.02)',
        'ErdosRenyi(N=500,p=0.03)',
        'WattsStrogatz(N=500,k=4,p=0.01)',
        'WattsStrogatz(N=500,k=4,p=0.02)',
        'WattsStrogatz(N=500,k=5,p=0.01)'
    )
    n_bins = 100  # 1000 should be 1 decimal point of precision for percentages

    csv_rows: List[Union[List[str], List[int], List[float]]] = []
    for class_ in classes:
        rng = np.random.default_rng(777)
        nets = fio.open_network_class(class_)
        communicabilities: List[int] = [sum(cell
                                            for row in nx.communicability_exp(net.G).values()
                                            for cell in row.values())
                                        for net in tqdm(nets, desc='Communicability')]
        entropies = [calc_entropy(run_sim_batch(net, 500, sd.Disease(4, .3),
                                                sd.SimplePressureBehavior(net, rng, 2, .25), rng),
                                  n_bins)
                     for net in tqdm(nets, 'Simulations & Entropy')]
        csv_rows.append(['Network Class', class_])
        csv_rows.append(['Communicability'])
        csv_rows.append(communicabilities)
        csv_rows.append(['Entropy'])
        csv_rows.append(entropies)

    with open(f'results/communicability-vs-infection-entropy-bins-{n_bins}.csv',
              'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_rows)


def save_simulation_data(n_sims: int, seed: int, pressure_radius: int, flicker_prob: float):
    classes = (
        'BarabasiAlbert(N=500,m=2)',
        'BarabasiAlbert(N=500,m=3)',
        'BarabasiAlbert(N=500,m=4)',
        'ConnComm(N_comm=10,ib=(5, 10),num_comms=50,ob=(3, 6))',
        'ConnComm(N_comm=20,ib=(15, 20),num_comms=25,ob=(3, 6))',
        'ErdosRenyi(N=500,p=0.01)',
        'ErdosRenyi(N=500,p=0.02)',
        'ErdosRenyi(N=500,p=0.03)',
        'WattsStrogatz(N=500,k=4,p=0.01)',
        'WattsStrogatz(N=500,k=4,p=0.02)',
        'WattsStrogatz(N=500,k=5,p=0.01)'
    )
    # This is only for getting the name. It'll get recreated for each class of network.
    behavior_config = SimplePressureConfig(pressure_radius, flicker_prob, None)
    disease = sd.Disease(4, .3)
    csv_rows: List[Union[List[str], np.ndarray]] = [
        ['A default_rng instance is created fresh for each set of simulations. '
         'It first gets passed into a SimplePressureConfig. Simulations were '
         'run one after the other with that same default_rng instance '
         'being repeatedly given to the simulate function. Each row of numbers are the '
         'survival rates of a particular instance of a network class for sum number of '
         'simulations.',
         f'seed = {seed}',
         f'sims = {n_sims}',
         f'behavior = {behavior_config.name}'
         f'disease = {disease}']
    ]
    for class_ in classes:
        print(f'Running simulations for {class_}')
        rng = np.random.default_rng(777)
        # recreate the config here
        behavior_config = SimplePressureConfig(pressure_radius, flicker_prob, rng)
        nets = fio.open_network_class(class_)
        csv_rows.append([class_])
        for net in tqdm(nets):
            behavior = behavior_config.make_behavior(net)
            survival_rates = run_sim_batch(net, n_sims, sd.Disease(4, .3),
                                           behavior, rng)
            csv_rows.append(survival_rates)

    with open(os.path.join('results', 'network-class-survival-rates.csv'),
              'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_rows)


def save_classes():
    """Save many instances of the networks in a gunzipped tarball."""
    n_instances = 500
    make_network_funcs = (
        MakeConnectedCommunity(20, (15, 20), 25, (3, 6)),
        MakeConnectedCommunity(10, (5, 10), 50, (3, 6)),
        MakeWattsStrogatz(500, 4, .01),
        MakeWattsStrogatz(500, 4, .02),
        MakeWattsStrogatz(500, 5, .01),
        MakeBarabasiAlbert(500, 2),
        MakeBarabasiAlbert(500, 3),
        MakeBarabasiAlbert(500, 4),
        MakeErdosRenyi(500, .01),
        MakeErdosRenyi(500, .02),
        MakeErdosRenyi(500, .03)
    )

    for make_network in tqdm(make_network_funcs):
        net_class = make_network.class_name
        networks = ((f'instance-{i}', make_network()) for i in range(n_instances))

        os.chdir('/tmp')
        if not os.path.exists(net_class):
            os.mkdir(net_class)
        for name, net in networks:
            fio.write_network(net.G, f'{net_class}/{name}', net.layout, net.communities)
        tar_name = f'{net_class}.tar.gz'
        with tarfile.open(tar_name, 'w:gz') as tar:
            tar.add(net_class)


if __name__ == '__main__':
    save_simulation_data(500, 777, 2, .25)
