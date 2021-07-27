from abc import ABC, abstractmethod
import sys
sys.path.append('')
from customtypes import Array
from collections import defaultdict
from common import (PressureComparisonResult, PressureConfig, RandomFlickerConfig,
                    simulate_return_survival_rate)
from typing import Dict, Sequence, Callable, List, Tuple
from sim_dynamic import Disease, make_starting_sir, no_update, simulate, PressureBehavior
from network import Network
from tqdm import tqdm
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkgen import make_connected_community_network
import fileio as fio

RNG = np.random.default_rng()


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


class MakeRandomNetwork(ABC):
    @property
    @abstractmethod
    def class_name(self) -> str:
        pass

    @abstractmethod
    def __call__(self) -> Network:
        pass


class MakeConnectedCommunity(MakeRandomNetwork):
    def __init__(self, community_size: int, inner_bounds: Tuple[int, int],
                 num_comms: int, outer_bounds: Tuple[int, int], rng):
        self._community_size = community_size
        self._inner_bounds = inner_bounds
        self._num_comms = num_comms
        self._outer_bounds = outer_bounds
        self._rng = rng
        self._class_name = f'ConnComm(N_comm={community_size},ib={inner_bounds},'\
                           f'num_comms={num_comms},ob={outer_bounds})'

    @property
    def class_name(self) -> str:
        return self._class_name

    def __call__(self) -> Network:
        id_dist = self._rng.integers(self._inner_bounds[0], self._inner_bounds[1],
                                     self._community_size, endpoint=True)
        if np.sum(id_dist) % 2 > 0:
            id_dist[np.argmin(id_dist)] += 1
        od_dist = self._rng.integers(self._outer_bounds[0], self._outer_bounds[1],
                                     self._num_comms, endpoint=True)
        if np.sum(od_dist) % 2 > 0:
            od_dist[np.argmin(od_dist)] += 1

        net = make_connected_community_network(id_dist, od_dist, self._rng)
        if net is None:
            raise Exception('This should not have happened.')
        return net


class MakeBarabasiAlbert(MakeRandomNetwork):
    def __init__(self, N: int, m: int, seed: int):
        self._N = N
        self._m = m
        self._seed = seed
        self._class_name = f'AlbertBarabasi(N={N},m={m})'

    @property
    def class_name(self) -> str:
        return self._class_name

    def __call__(self) -> Network:
        return Network(nx.barabasi_albert_graph(self._N, self._m, self._seed))


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
    G, layout, communities = fio.read_network('networks/cavemen-10-10.txt')
    if layout is None or communities is None:
        raise Exception('File is incomplete.')
    net = Network(G, communities=communities)
    simulate(net.M, make_starting_sir(net.N, (0,)), Disease(4, 0.3),
             PressureBehavior(net), 200, layout, RNG)


def pressure_experiment(make_network: MakeRandomNetwork,
                        pressure_configurations: Sequence[PressureConfig],
                        disease: Disease, num_trials: int, rng) -> None:
    pressure_type_to_survival_rates = {}
    static_survival_rates = np.array([simulate_return_survival_rate(make_network(), disease,
                                                                    no_update, rng)
                                      for _ in range(num_trials)])
    pressure_type_to_survival_rates['Static'] = static_survival_rates

    for configuration in pressure_configurations:
        networks = [make_network() for _ in range(num_trials)]
        behaviors = [configuration.make_behavior(net) for net in networks]
        pressure_type_to_survival_rates[behaviors[0].name]\
            = np.array([simulate_return_survival_rate(net, disease, behavior, rng)
                        for net, behavior in zip(networks, behaviors)])

    result = PressureComparisonResult(make_network.class_name, disease, num_trials,
                                      pressure_type_to_survival_rates, 'Static')
    result.save('results', True)
    result.save_raw('results')


def cc_pressure_vs_none_entry_point():
    rng = np.random.default_rng(0xbeefee)
    num_trials = 250
    disease = Disease(4, .4)
    inner_bounds = 1, 15
    outer_bounds = 1, 5
    community_size = 20
    n_communities = 25
    make_ccn = MakeConnectedCommunity(community_size, inner_bounds, n_communities,
                                      outer_bounds, rng)
    pressure_configurations = (PressureConfig(1, .75, rng), PressureConfig(1, .25, rng),
                               PressureConfig(3, .75, rng), PressureConfig(3, .25, rng))
    pressure_experiment(make_ccn, pressure_configurations, disease, num_trials, rng)


def ba_pressure_vs_none_entry_point():
    rng = np.random.default_rng(0xbeefee)
    num_trials = 250
    disease = Disease(4, .4)
    N = 500
    m = 3
    make_ba = MakeBarabasiAlbert(N, m, 0xbeefee)
    pressure_configurations = [PressureConfig(radius, prob, rng)
                               for radius, prob in it.product((1, 2, 3), (.25, .5, .75))]
    pressure_experiment(make_ba, pressure_configurations, disease, num_trials, rng)


if __name__ == '__main__':
    # cc_pressure_vs_none_entry_point()
    ba_pressure_vs_none_entry_point()
