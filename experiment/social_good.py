import sys
sys.path.append('')
from customtypes import Array, Number
from common import MakeConnectedCommunity
from typing import Callable, Sequence, Tuple
from network import Network
from socialgood import DecayFunction, rate_social_good
from tqdm import tqdm
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


def main():
    ks = (.25, .4, .5, .6, .75, 1.0, 1.5, 2.0)[:2]
    for k in ks:
        erdos_renyi_k_experiment(k)
        # watts_strogatz_k_experiment(k)


def k_experiment(k: float, name: str, n_trials: int,
                 independent_variable: Array, make_network: Callable[[Number], Network]):
    decay = DecayFunction(k)
    all_stats = np.zeros((len(independent_variable), n_trials))
    for i, x in tqdm(tuple(enumerate(independent_variable))):
        stats = np.array(list(map(rate_sg, [(make_network(x), decay)
                                            for _ in tqdm(range(n_trials))])))
        # with Pool(4) as pool:
        #     stats = np.array(pool.map(rate_sg,
        #                               [(Network(nx.erdos_renyi_graph(N, p)), decay)
        #                                for _ in range(n_trials)],
        #                               n_trials//4))
        all_stats[i] = stats

    quartiles = np.quantile(all_stats, (.25, .75), axis=1, interpolation='midpoint')
    plt.clf()
    plt.title(name)
    line = np.mean(all_stats, axis=1)
    plt.plot(independent_variable, line)
    plt.fill_between(independent_variable, quartiles[0], quartiles[1], alpha=.4)
    plt.savefig(f'results/{name}.png', format='png', dpi=300)


def erdos_renyi_k_experiment(k: float):
    N = 500
    n_divisions = 20
    n_trials = 250
    ps = np.linspace(0, 1, n_divisions)
    name = f'Social Good of Random Networks with p=0..1 k={k}'
    k_experiment(k, name, n_trials, ps, lambda p: Network(nx.erdos_renyi_graph(N, p)))


def cc_social_good_variance_entry_point():
    rng = np.random.default_rng(404)
    num_trials = 250
    inner_bounds = 1, 15
    community_size = 20
    n_comms = 25

    def make_ccn(outer_connections: Number):
        return MakeConnectedCommunity(community_size, inner_bounds, n_comms,
                                      (outer_connections, outer_connections), rng)()  # type: ignore

    k_experiment(.5,
                 f'ConnComm(N_comm={community_size},ib={inner_bounds},num_comms={n_comms},ob=x)',
                 num_trials, range(1, 11), make_ccn)


def watts_strogatz_k_experiment(decay_coeff: float):
    N = 500
    n_divisions = 20
    n_trials = 250
    rewiring_prob = .01
    approximate_edge_densities = np.linspace(0, 1, n_divisions)
    decay = DecayFunction(decay_coeff)
    nearest_neighbor_values = [k_from_approx_edge_density(N, aed)
                               for aed in approximate_edge_densities]
    max_edges = N*(N-1)//2
    # This is filled up in the main loop
    actual_edge_densities = np.zeros(approximate_edge_densities.shape)
    name = f'Social Good of Watts-Strogatz with k={decay_coeff}'
    all_stats = np.zeros((n_divisions, n_trials))
    for i, k in tqdm(tuple(enumerate(nearest_neighbor_values))):
        # if k == 0, connected_watts_strogatz_graph will throw an exception and
        # social good will be 0
        if k == 0:
            stats = np.zeros(n_trials)
            actual_edge_densities[i] = 0.0
        else:
            networks = (nx.connected_watts_strogatz_graph(N, k, rewiring_prob)
                        for _ in range(n_trials))
            networks = [Network(G) for G in networks]
            actual_edge_densities[i] = len(networks[0].G.edges) / max_edges
            with Pool(4) as pool:
                stats = np.array(pool.map(rate_sg,
                                          [(net, decay) for net in networks],
                                          n_trials//4))
        all_stats[i] = stats

    quartiles = np.quantile(all_stats, (.25, .75), axis=1, interpolation='midpoint')
    plt.clf()
    plt.title(name)
    line = np.mean(all_stats, axis=1)
    plt.xlabel('Edge Density')
    plt.ylabel('Social Good Score')
    plt.plot(actual_edge_densities, line)
    plt.fill_between(actual_edge_densities, quartiles[0], quartiles[1], alpha=.4)
    plt.savefig(f'results/{name}.png', format='png', dpi=300)


def k_from_approx_edge_density(N: int, approx_edge_density: float) -> int:
    """
    Return how many neighbors a node in a Watts-Strogatz network must connect
    with in order to get as close to the approximate edge density as possible.
    """
    try:
        k = int(np.round(approx_edge_density*(N-1)))
    except OverflowError:
        k = N // 2 + 1
    return k


def rate_sg(args: Tuple[Network, DecayFunction])\
        -> float:
    net, decay = args
    return rate_social_good(net, decay)


if __name__ == '__main__':
    try:
        main()
    except EOFError:
        print('\nGood-bye.')
    except KeyboardInterrupt:
        print('\nGood-bye.')
