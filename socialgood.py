from customtypes import Number
from typing import Callable, Generic, Sequence, TypeVar
import numpy as np
import networkx as nx
import fileio as fio
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from analyzer import visualize_network
from networkgen import _connected_community as cc
import networkx as nx
from matplotlib import pyplot as plt
from network import Network
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


def get_distance_matrix(net: Network) -> np.ndarray:
    """
    Returns the distance matrix of a given matrix with infinity value given.
    """
    if nx.is_connected(net.G):
        # I expect this will be faster than our algorithm -- especially for
        # any student interaction network.
        return nx.floyd_warshall_numpy(net.G)

    M = net.M
    num_nodes = len(M)
    m = np.copy(M)
    dm = np.copy(M)
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

    # Sets the given infinity value before return.
    for n in range(num_nodes):
        dm[n, n] = np.inf
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
                G, _, _ = fio.read_network(path)
                net = Network(G)
                social_good_score = rate_social_good(net, decay_func)
                scores.append(f'{social_good_score:.3f}')
            writer.writerow([name] + scores)


def visualize_social_good(networks: Sequence[str], network_paths: Sequence[str]):
    for name, path in zip(networks, network_paths):
        G, layout, _ = fio.read_network(path)
        node_size = node_size_from_social_good(G, DecayFunction(1))
        plt.title(f'{name} Node Size')
        plt.hist(node_size, bins=None)
        plt.figure()
        print(f'{name} min = {np.min(node_size):.2f} max = {np.max(node_size):.2f}')
        visualize_network(G, layout, name, node_size=node_size, block=False)

    input('Done.')


def main():
    # networks = ('cavemen-50-10', 'elitist-500', 'agent-generated-500',
    #             'annealed-agent-generated-500', 'barabasi-albert-500-3', 'cgg-500',
    #             'connected-comm-50-10', 'spatial-network', 'watts-strogatz-500-4-.1')
    # network_paths = fio.network_names_to_paths(networks)
    # visualize_social_good(networks, network_paths)

    # RAND = np.random.default_rng()

    # outf = open('social-good-id0:20-od0:10_actual_degrees.txt', 'w+')
    # num_id = 20
    # num_od = 10
    # # avg_social_goods = np.zeros((num_id, num_od))
    # for i in range(num_id):
    #     for j in range(num_od):
    #         social_goods = []
    #         deg_dists = []
    #         print(f'i: {i+1}/{num_id}, j: {j+1}/{num_od}')
    #         for n in range(100):
    #             inner_degrees = np.round(RAND.poisson(i, 20))
    #             if np.sum(inner_degrees) % 2 == 1:
    #                 inner_degrees[np.argmin(inner_degrees)] += 1
    #             outer_degrees = np.round(RAND.poisson(i, 10))
    #             if np.sum(outer_degrees) % 2 == 1:
    #                 outer_degrees[np.argmin(outer_degrees)] += 1
    #             graph, _ = cc.make_connected_community_network(inner_degrees, outer_degrees, RAND)
    #             deg_dist = [d for _, d in graph.degree()]
    #             deg_dists.append(sum(deg_dist) / len(deg_dist))
    #             social_goods.append(rate_social_good(graph))
    #         avg_social_good = sum(social_goods) / len(social_goods)
    #         # avg_social_goods[i, j] = avg_social_good
    #         outf.write(f'{i} {j} {avg_social_good:.4f} {sum(deg_dists) / len(deg_dists)} {min(social_goods)} {max(social_goods)}\n')
    # outf.close()

    # np.set_printoptions(precision=4)
    # for i in range(num_od):
    #     x = range(len(avg_social_goods))
    #     y = avg_social_goods[:, i]
    #     plt.plot(x, y, 'o', color='black')
    # plt.xlabel('Average Inner Degree')
    # plt.ylabel('Average Social Good')
    # plt.show()
    # print(np.array(avg_social_goods))
    # plt.title(f'avg_social_goods: min = {min(avg_social_goods)}, max = {max(avg_social_goods)}')
    # plt.hist(avg_social_goods)
    # plt.show()
    # nx.draw(graph)
    # plt.show()
    networks = ('cavemen-50-10', 'elitist-500', 'agent-generated-500',
                'annealed-agent-generated-500', 'barabasi-albert-500-3', 'cgg-500',
                'connected-comm-50-10', 'spatial-network', 'watts-strogatz-500-4-.1')
    network_paths = fio.network_names_to_paths(networks)
    for name, path in zip(networks, network_paths):
        G, _, _ = fio.read_network(path)
        net = Network(G)
        print(f'{name:<30} {rate_social_good(net, DecayFunction(.5)):>10.3f}')


if __name__ == '__main__':
    try:
        main()
    except EOFError:
        print('\nGood-bye.')
    except KeyboardInterrupt:
        print('\nGood-bye.')
