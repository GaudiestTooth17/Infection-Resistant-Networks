import sys
sys.path.append('')
from collections import defaultdict
from common import (MakeBarabasiAlbert, MakeConnectedCommunity, MakeNetwork,
                    PressureComparisonResult, SimplePressureConfig, RandomFlickerConfig,
                    simulate_return_survival_rate)
from typing import Dict, Sequence, Callable, List
from sim_dynamic import Disease, make_starting_sir, no_update, simulate, SimplePressureBehavior
from network import Network
from tqdm import tqdm
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import fileio as fio
import csv


def run_inf_prob_vs_perc_sus(name: str, diseases: Sequence[Disease],
                             new_network: Callable[[], Network],
                             flicker_config: RandomFlickerConfig,
                             num_trials: int, rng):
    """
    Save the plot and csv of infection probability vs percent susceptible.
    """
    results: Dict[float, List[float]] = defaultdict(lambda: [])
    print(f'Running {name}')
    pbar = tqdm(total=num_trials*len(diseases))
    for disease, _ in it.product(diseases, range(num_trials)):
        net = new_network()
        flicker = flicker_config.make_behavior(net.M, net.intercommunity_edges)
        sir0 = make_starting_sir(net.N, 1, rng)
        perc_sus = np.sum(simulate(net.M, sir0, disease, flicker, 100, None, rng)[-1][0] > 0)/net.N
        results[disease.trans_prob].append(perc_sus)
        pbar.update()

    x_coords = tuple(results.keys())
    collected_data = np.array([list(values) for values in results.values()])
    quartiles = np.quantile(collected_data, (.25, .75), axis=1, interpolation='midpoint')
    y_coords = np.mean(collected_data, axis=1)
    plt.figure()
    plt.title(name)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Infection Probability')
    plt.ylabel('Survival Percentage')
    plt.plot(x_coords, y_coords)
    plt.fill_between(x_coords, quartiles[0], quartiles[1], alpha=.4)
    plt.savefig(f'results/{name}.png', format='png', dpi=200)


def test():
    name = 'test'
    diseases = [Disease(1, .69)]

    def new_network():
        return Network(nx.connected_caveman_graph(10, 10), community_size=10)

    flicker_config = RandomFlickerConfig(.5, 'test_rand_config')
    rng = np.random.default_rng(66)
    run_inf_prob_vs_perc_sus(name, diseases, new_network, flicker_config, 10, rng)


def connected_community_entry_point():
    rng = np.random.default_rng(501)
    min_inner, max_inner = 1, 15
    min_outer, max_outer = 1, 5
    community_size = 20
    n_communities = 25
    next_network = MakeConnectedCommunity(community_size, (min_inner, max_inner),
                                          n_communities, (min_outer, max_outer), rng)
    base_name = f'Connected Community N_comm={community_size} [{min_inner}, {max_inner}]\n'\
        f'num_comms={n_communities} [{min_outer}, {max_outer}]'
    diseases = [Disease(2, trans_prob) for trans_prob in np.linspace(.1, 1.0, 25)]
    for flicker_prob in np.linspace(1, .2, 9):
        run_inf_prob_vs_perc_sus(f'{base_name} flicker_prob={flicker_prob:.2f}',
                                 diseases, next_network,
                                 RandomFlickerConfig(flicker_prob, rand=rng), 100, rng)


def pressure_test_entry_point():
    rng = np.random.default_rng()
    net = fio.read_network('networks/cavemen-50-10.txt')
    simulate(net.M, make_starting_sir(net.N, (0,), rng), Disease(4, 0.3),
             SimplePressureBehavior(net, 1), 200, None, rng)


def pressure_experiment(make_network: MakeNetwork,
                        pressure_configurations: Sequence[SimplePressureConfig],
                        disease: Disease, num_trials: int, rng) -> None:
    pressure_type_to_survival_rates = {}
    static_survival_rates = np.array([simulate_return_survival_rate(make_network(), disease,
                                                                    no_update, rng)
                                      for _ in range(num_trials)])
    pressure_type_to_survival_rates['Static'] = static_survival_rates

    pbar = tqdm(desc='Pressure Experiment',
                total=len(pressure_configurations)*num_trials*3)

    def simulate_and_update(net, disease, behavior, rng):
        result = simulate_return_survival_rate(net, disease, behavior, rng)
        pbar.update()
        return result

    def make_net_and_update():
        net = make_network()
        pbar.update()
        return net

    def make_behavior_and_update(net: Network):
        behavior = configuration.make_behavior(net)
        pbar.update()
        return behavior

    for configuration in pressure_configurations:
        networks = [make_net_and_update() for _ in range(num_trials)]
        behaviors = [make_behavior_and_update(net) for net in networks]
        pressure_type_to_survival_rates[behaviors[0].name]\
            = np.array([simulate_and_update(net, disease, behavior, rng)
                        for net, behavior in zip(networks, behaviors)])

    result = PressureComparisonResult(make_network.class_name, disease, num_trials,
                                      pressure_type_to_survival_rates, 'Static')
    result.save('results', True)
    result.save_raw('results')


def cc_pressure_vs_none_entry_point():
    rng = np.random.default_rng(0xbeefee)
    num_trials = 1000
    disease = Disease(4, .4)
    inner_bounds = 1, 15
    outer_bounds = 1, 5
    community_size = 20
    n_communities = 25
    make_ccn = MakeConnectedCommunity(community_size, inner_bounds, n_communities,
                                      outer_bounds, rng)
    pressure_configurations = [SimplePressureConfig(radius, prob, rng)
                               for radius, prob in it.product((1, 2, 3), (.25, .5, .75))]
    # pressure_experiment(make_ccn, pressure_configurations, disease, num_trials, rng)
    net = make_ccn()
    simulate(net.M, make_starting_sir(net.N, 1, rng), disease,
             pressure_configurations[1].make_behavior(net),
             100, nx.spring_layout(net.G))


def ba_pressure_vs_none_entry_point():
    rng = np.random.default_rng(0xbeefee)
    num_trials = 1000
    disease = Disease(4, .4)
    N = 500
    m = 3
    make_ba = MakeBarabasiAlbert(N, m, 0xbeefee)
    pressure_configurations = [SimplePressureConfig(radius, prob, rng)
                               for radius, prob in it.product((1, 2, 3), (.25, .5, .75))]
    pressure_experiment(make_ba, pressure_configurations, disease, num_trials, rng)


def save_communicabilities():
    network_classes = (
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
    rows = []
    for class_name in network_classes:
        nets = fio.read_network_class(class_name)
        communicabilties = (sum(c
                                for inner_values in nx.communicability_exp(net.G).values()
                                for c in inner_values.values())
                            for net in nets)
        rows.append([class_name])
        rows.append(list(map(str, communicabilties)))

    with open('communicabilities.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(rows)


if __name__ == '__main__':
    pressure_test_entry_point()
