from customtypes import Number
from typing import Callable, Generic, Sequence, TypeVar
import numpy as np
import networkx as nx
import fileio as fio
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from analysis import visualize_network
import networkx as nx
from matplotlib import pyplot as plt
from network import Network
import retworkx as rx
import time
T = TypeVar('T', Number, np.ndarray)
TDecayFunc = Callable[[T], T]


class DecayFunction(Generic[T]):
    function_desc = '1/(distance^k)'

    def __init__(self, k: Number):
        """Return a social good value based on how far away two nodes are."""
        self.k = k

    def __call__(self, distance: T) -> T:
        with np.errstate(divide='ignore'):
            result = 1/(distance**self.k)
        result = np.where(result == np.inf, 0, result)
        return result


def get_distance_matrix_deprecated(net: Network) -> np.ndarray:
    """
    Returns the distance matrix of a given matrix with infinity value given.
    """
    # I profiled the code while running this on Erdos-Renyi networks, and found
    # out that the code runs faster without this if block. It's possible this
    # doesn't apply with other classes of network.
    # if nx.is_connected(net.G):
    #     # I expect this will be faster than our algorithm -- especially for
    #     # any student interaction network.
    #     return nx.floyd_warshall_numpy(net.G)

    M = net.M
    num_nodes = len(M)
    m = np.copy(M)
    dm = np.copy(M).astype(np.float64)
    dm[dm < 1] = np.inf
    x = np.copy(M)

    for d in range(num_nodes):
        old_x = x
        x = x @ m
        x[x > 0] = 1
        if (old_x == x).all():
            break

        # For every new path we know that the distance is d + 2 since
        #  d starts at 0 and we already have everything of distance 1.
        dm[np.logical_and(dm == np.inf, x != 0)] = d + 2

    # Sets the self distance to zero before return.
    for n in range(num_nodes):
        dm[n, n] = 0
    return dm


def get_distance_matrix(net: Network) -> np.ndarray:
    dm: np.ndarray = rx.distance_matrix(net.R).copy()  # type: ignore
    dm[dm == 0] = np.inf
    for u in range(len(dm)):
        dm[u, u] = 0
    return dm


def rate_social_good(net: Network,
                     decay_func: TDecayFunc = DecayFunction(1)) -> float:
    """
    Rate a network on how much social good it has.
    """

    N = net.N
    # If there is only 1 node, the score is 0.
    if N == 1:
        return 0
    dist_matrix = get_distance_matrix(net)
    # Sets self distance to infinity to avoid counting itself for social good
    for n in range(N):
        dist_matrix[n, n] = np.inf

    social_good_scores = decay_func(dist_matrix)
    social_good_scores[social_good_scores == np.inf] = 0
    return np.sum(social_good_scores) / (N*(N-1))


def node_size_from_social_good(G: nx.Graph, decay_func: TDecayFunc) -> Sequence[Number]:
    dist_matrix = get_distance_matrix(Network(G))

    def calc_score(u: int, v: int) -> float:
        return 0 if u == v else decay_func(dist_matrix[u][v])

    social_good_scores = np.array(tuple(tuple(calc_score(u, v) for v in G.nodes) for u in G.nodes))
    node_size = np.array([np.sum(social_good_scores[u])+np.sum(social_good_scores[:, u])
                          for u in G.nodes]) / len(G) * 100
    return node_size


def save_social_good_csv(networks: Sequence[str], network_paths: Sequence[str]):
    decay_functions = (DecayFunction(.5), DecayFunction(1), DecayFunction(2))

    with open('social-good.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([DecayFunction.function_desc] + [str(df.k) for df in decay_functions])

        for name, path in tqdm(tuple(zip(networks, network_paths))):
            scores = []
            for decay_func in decay_functions:
                net = fio.read_network(path)
                social_good_score = rate_social_good(net, decay_func)
                scores.append(f'{social_good_score:.3f}')
            writer.writerow([name] + scores)


def visualize_social_good(networks: Sequence[str], network_paths: Sequence[str]):
    for name, path in zip(networks, network_paths):
        net = fio.read_network(path)
        node_size = node_size_from_social_good(net.G, DecayFunction(1))
        plt.title(f'{name} Node Size')
        plt.hist(node_size, bins=None)
        plt.figure()
        print(f'{name} min = {np.min(node_size):.2f} max = {np.max(node_size):.2f}')
        visualize_network(net.G, net.layout, name, node_size=node_size, block=False)

    input('Done.')


def main():
    networks = ('cavemen-50-10', 'elitist-500', 'agent-generated-500',
                'annealed-agent-generated-500', 'barabasi-albert-500-3', 'cgg-500',
                'connected-comm-50-10', 'spatial-network', 'watts-strogatz-500-4-.1')
    network_paths = fio.network_names_to_paths(networks)
    for name, path in zip(networks, network_paths):
        net = fio.read_network(path)
        print(f'{name:<30} {rate_social_good(net, DecayFunction(.5)):>10.3f}')


def speed_test():
    # net = Network(nx.grid_2d_graph(100, 100))
    net = Network(nx.caveman_graph(100, 100))
    start_time = time.time()
    dm = get_distance_matrix(net)
    print(f'Finished retworkx implementation ({time.time()-start_time} s)')
    print(dm[0, 200])


if __name__ == '__main__':
    try:
        main()
    except EOFError:
        print('\nGood-bye.')
    except KeyboardInterrupt:
        print('\nGood-bye.')
