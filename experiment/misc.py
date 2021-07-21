import sys
sys.path.append('')
from collections import defaultdict
from common import RandomFlickerConfig
from typing import Dict, Sequence, Callable, List, Tuple
from sim_dynamic import Disease, make_starting_sir, simulate
from network import Network
from tqdm import tqdm
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkgen import make_connected_community_network


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


class MakeConnectedCommunity:
    def __init__(self, community_size: int, inner_bounds: Tuple[int, int],
                 num_comms: int, outer_bounds: Tuple[int, int], rng):
        self._community_size = community_size
        self._inner_bounds = inner_bounds
        self._num_comms = num_comms
        self._outer_bounds = outer_bounds
        self._rng = rng

    def __call__(self) -> Network:
        id_dist = self._rng.integers(self._inner_bounds[0], self._inner_bounds[1],
                                     self._community_size, endpoint=True)
        if np.sum(id_dist) % 2 > 0:
            id_dist[np.argmin(id_dist)] += 1
        od_dist = self._rng.integers(self._outer_bounds[0], self._outer_bounds[1],
                                     self._num_comms, endpoint=True)
        if np.sum(od_dist) % 2 > 0:
            od_dist[np.argmin(od_dist)] += 1

        result = make_connected_community_network(id_dist, od_dist, self._rng)
        if result is None:
            raise Exception('This should not have happened.')
        G, communities = result
        return Network(G, communities=communities)


def connected_community_entry_point():
    rng = np.random.default_rng(501)
    min_inner, max_inner = 1, 15
    min_outer, max_outer = 5, 10
    community_size = 20
    n_communities = 25
    next_network = MakeConnectedCommunity(community_size, (min_inner, max_inner),
                                          n_communities, (min_outer, max_outer), rng)
    base_name = f'Connected Community N_comm={community_size} [{min_inner}, {max_inner}]\n'\
        f'num_comms={n_communities} [{min_outer}, {max_outer}]'
    diseases = [Disease(2, trans_prob) for trans_prob in np.linspace(.1, 1.0, 10)]
    for flicker_prob in np.linspace(1, .2, 9):
        run_inf_prob_vs_perc_sus(f'{base_name} flicker_prob={flicker_prob:.2f}',
                                 diseases, next_network,
                                 RandomFlickerConfig(flicker_prob, rand=rng), 100, rng)


if __name__ == '__main__':
    connected_community_entry_point()
