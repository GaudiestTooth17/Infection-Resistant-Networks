import sys
sys.path.append('')
from network import Network
from socialgood import get_distance_matrix
from typing import Any, Callable, List, Sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkgen import MakeLazySpatialNetwork, make_random_spatial_configuration
from sim_dynamic import (Disease, SimplePressureBehavior, UpdateConnections,
                         simulate, make_starting_sir, no_update)
from common import calc_survival_rate, RawDataCSV
import itertools as it


class SpatialContraction:
    def __init__(self, normal_net: Network,
                 contracted_net: Network,
                 pressure_radius: int,
                 flicker_prob: float,
                 rng):
        self._normal_net = normal_net
        self._contracted_net = contracted_net
        self._pressure_radius = pressure_radius
        self._flicker_prob = flicker_prob
        self._rng = rng
        # TODO: consider replacing this with the distance matrix from a MakeLazySpatial instance
        # because that matrix represents how close "socially" two people are.
        # The downside to using that is that then the behavior won't correspond as closely
        # to SimplePressureBehavior
        self._dm = get_distance_matrix(normal_net)
        self._pressure = np.zeros(normal_net.N, dtype=np.uint32)

    def __call__(self, D: np.ndarray, M: np.ndarray, time_step: int, sir: np.ndarray) -> np.ndarray:
        infectious_agents = sir[1] == 1
        if infectious_agents.any():
            pressured_agents = (self._dm[infectious_agents] <= self._pressure_radius)[0]
            self._pressure[pressured_agents] += 1

        recovered_agents = sir[2] == 1
        if recovered_agents.any():
            unpressured_agents = (self._dm[recovered_agents] <= self._pressure_radius)[0]
            self._pressure[unpressured_agents] -= 1

        flicker_agents = ((self._pressure > 0) & (self._rng.random(self._pressure.shape)
                                                  < self._flicker_prob))
        R = np.copy(M)
        R[flicker_agents, :] = self._contracted_net.M[flicker_agents, :]
        R[:, flicker_agents] = self._contracted_net.M[:, flicker_agents]
        return R


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
    plt.show()


def two_reach_sim(lazy_networks: Sequence[MakeLazySpatialNetwork], sims_per_network: int,
                  disease: Disease, high_reach: int, low_reach: int, sim_len_cap: int,
                  make_behavior: Callable[[int, MakeLazySpatialNetwork, Any], UpdateConnections],
                  behavior_name: str) -> RawDataCSV:

    def experiment(make_network, seed):
        """
        return: list of (high_reach_survival_rate, high_reach_sim_time,
                         low_reach_survival_rate, low_reach_sim_time)
        """
        rng = np.random.default_rng(seed)
        hr_net = make_network(high_reach)
        lr_net = make_network(low_reach)
        sir0 = make_starting_sir(hr_net.N, 1, rng)
        hr_sirs = simulate(hr_net.M, sir0, disease, make_behavior(high_reach, make_network, rng),
                           sim_len_cap, rng, None)
        lr_sirs = simulate(lr_net.M, sir0, disease, make_behavior(low_reach, make_network, rng),
                           sim_len_cap, rng, None)
        return (calc_survival_rate(hr_sirs), len(hr_sirs),
                calc_survival_rate(lr_sirs), len(lr_sirs))

    data = [experiment(make_network, seed)
            for make_network, seed
            in tqdm(tuple(it.product(lazy_networks, range(sims_per_network))))]
    hr_survival_rates, hr_sim_times, lr_survival_rates, lr_sim_times = zip(*data)

    distributions = {
        f'Reach {high_reach} {behavior_name} Survival Rates': hr_survival_rates,
        f'Reach {high_reach} {behavior_name} Simulation Times': hr_sim_times,
        f'Reach {low_reach} {behavior_name} Survival Rates': lr_survival_rates,
        f'Reach {low_reach} {behavior_name} Simulation Times': lr_sim_times
    }
    return RawDataCSV(f'spatial-nets-sim_len={sim_len_cap}-{disease}',
                      distributions)  # type: ignore


def generate_all_data_for_two_reach():
    num_networks = 1000
    sims_per_network = 25
    high_reach = 60
    low_reach = 30
    contracted_reach = 20
    sim_len_cap = 1000
    disease = Disease(4, .3)
    lazy_networks = [
        MakeLazySpatialNetwork(make_random_spatial_configuration((500, 500), 500,
                                                                 np.random.default_rng(seed)))
        for seed in tqdm(range(num_networks))
    ]

    # TODO: add flicker and run
    behaviors = ((lambda reach, mknet, rng: no_update, 'Static Network'),
                 (lambda reach, mknet, rng: SimplePressureBehavior(mknet(reach), rng, 2, .5),
                  'SimplePressure(radius=2, flicker_prob=.5)'),
                 (lambda reach, mknet, rng: SpatialContraction(mknet(reach),
                                                               mknet(contracted_reach),
                                                               2, .5, rng),
                  'SpatialContraction(radius=2, contraction_prob=.5)'))
    all_data = RawDataCSV(f'spatial nets\nsim_len={sim_len_cap} {disease}', {})
    for mkbeh, beh_name in behaviors:
        temp_data = two_reach_sim(lazy_networks, sims_per_network, disease, high_reach,
                                  low_reach, sim_len_cap, mkbeh, beh_name)
        all_data = RawDataCSV.union(all_data.title, all_data, temp_data)
    all_data.save().save_boxplots()


if __name__ == '__main__':
    try:
        generate_all_data_for_two_reach()
    except KeyboardInterrupt:
        print('\nGood bye.')
    except EOFError:
        print('\nGood bye.')
