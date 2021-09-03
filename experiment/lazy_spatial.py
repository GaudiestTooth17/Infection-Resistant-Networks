import sys
sys.path.append('')
from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkgen import MakeLazySpatialNetwork, make_random_spatial_configuration
from sim_dynamic import Disease, simulate, make_starting_sir, no_update
from common import calc_survival_rate
import itertools as it


def sensitivity_to_initial_configuration():
    """Test network creation speed and sensitivity of number of components to grid configuration."""
    reaches = np.linspace(1, 100, 100)
    all_num_comps: List[List[int]] = []
    all_min_degrees = []
    all_avg_degrees = []
    all_max_degrees = []
    num_seeds = 50
    for seed in tqdm(range(num_seeds)):
        configuration = make_random_spatial_configuration((500, 500), 500,
                                                          np.random.default_rng(seed))
        make_network = MakeLazySpatialNetwork(configuration)
        networks = [make_network(reach) for reach in reaches]
        num_comps = [nx.number_connected_components(network.G) for network in networks]
        all_num_comps.append(num_comps)
        degrees = [tuple(dict(net.G.degree).values()) for net in networks]
        all_min_degrees.append([min(d) for d in degrees])
        all_avg_degrees.append([np.average(d) for d in degrees])
        all_max_degrees.append([max(d) for d in degrees])

    _, axs = plt.subplots(2)
    ax1, ax2 = axs[0], axs[1]
    ps = [all_num_comps, all_min_degrees, all_avg_degrees, all_max_degrees]
    for i, p in enumerate(ps):
        num_comp_data = np.array(p)
        quartiles = np.quantile(num_comp_data, (.25, .75), axis=0, interpolation='midpoint')
        y_coords = np.mean(num_comp_data, axis=0)
        # plt.figure()
        # plt.title(str(i))
        if i == 0:
            ax1.set(ylabel='Num Components')
            # ax1.xlabel('Reach')
            # ax1.ylabel('Number of Components')
            ax1.plot(reaches, y_coords)
            ax1.fill_between(reaches, quartiles[0], quartiles[1], alpha=.4)
        else:
            ax2.set(xlabel='Reach', ylabel='Degree')
            # ax2.xlabel('Reach')
            # ax2.ylabel('Degree')
            ax2.plot(reaches, y_coords)
            ax2.fill_between(reaches, quartiles[0], quartiles[1], alpha=.4)
    # plt.plot(all_min_degrees)
    # plt.plot(all_avg_degrees)
    # plt.plot(all_max_degrees)
    plt.show()


def two_reach_survival_rates():
    lazy_networks = [
        MakeLazySpatialNetwork(make_random_spatial_configuration((500, 500), 500,
                                                                 np.random.default_rng(seed)))
        for seed in tqdm(range(100))
    ]
    high_reach = 60
    low_reach = 30
    disease = Disease(4, .2)
    # list of (high_reach_survival_rate, high_reach_sim_time,
    #          low_reach_survival_rate, low_reach_sim_time)

    def experiment(make_network):
        rng = np.random.default_rng(0)
        hr_net = make_network(high_reach)
        lr_net = make_network(low_reach)
        sir0 = make_starting_sir(hr_net.N, 1, rng)
        hr_sirs = simulate(hr_net.M, sir0, disease, no_update, 300, rng, None)
        lr_sirs = simulate(lr_net.M, sir0, disease, no_update, 300, rng, None)
        return (calc_survival_rate(hr_sirs), len(hr_sirs),
                calc_survival_rate(lr_sirs), len(lr_sirs))

    data = [experiment(make_network) for make_network in tqdm(lazy_networks)]
    hr_survival_rates, hr_sim_times, lr_survival_rates, lr_sim_times = zip(*data)

    def plot_hist(data, title, y_max):
        plt.ylim((0, y_max))
        plt.title(title)
        plt.hist(data, bins=None)
        plt.figure()

    sim_time_max = max(it.chain(hr_sim_times, lr_sim_times))
    plot_hist(hr_survival_rates, 'High Reach Survival Rates', 1)
    plot_hist(hr_sim_times, 'High Reach Sim Times', sim_time_max)
    plot_hist(lr_survival_rates, 'Low Reach Survival Rates', 1)
    plot_hist(lr_sim_times, 'Low Reach Sim Times', sim_time_max)
    plt.show()


if __name__ == '__main__':
    try:
        two_reach_survival_rates()
    except KeyboardInterrupt:
        print('\nGood bye.')
    except EOFError:
        print('\nGood bye.')
