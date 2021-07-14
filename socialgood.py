from customtypes import Number
from typing import Callable, Sequence, Set, Tuple
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
TDecayFunc = Callable[[int], Number]


class DecayFunction:
    function_desc = '1/(distance^k)'

    def __init__(self, k: Number):
        """Return a social good value based on how far away two nodes are."""
        self.k = k

    def __call__(self, distance: int) -> float:
        return 1/(distance**self.k)


def rate_social_good(G: nx.Graph, decay_func: TDecayFunc = DecayFunction(1)) -> float:
    """
    Rate a network on how much social good it has.
    """
    # The algorithm gets messed up if it runs on a single node network
    # and the correct answer is 0 anways
    if len(G) == 1:
        return 0

    components = tuple(nx.connected_components(G))
    # Floyd-Warshall doesn't work on unconnected networks, so we have to run
    # this function on each component separately.
    if len(components) > 1:
        social_good = 0
        for component in components:
            weight = len(component) / len(G)
            H = G.subgraph(component).copy()
            H = nx.relabel_nodes(H, mapping={old: new for new, old in enumerate(H.nodes)}, copy=True)
            social_good += rate_social_good(H, decay_func) * weight
        return social_good

    dist_matrix = nx.floyd_warshall_numpy(G)

    def calc_score(u: int, v: int) -> float:
        return 0 if u == v else decay_func(dist_matrix[u][v])

    social_good_scores = tuple(tuple(calc_score(u, v) for v in G.nodes) for u in G.nodes)
    return np.sum(social_good_scores) / (len(G)*(len(G)-1))


def node_size_from_social_good(G: nx.Graph, decay_func: TDecayFunc) -> Sequence[Number]:
    dist_matrix = nx.floyd_warshall_numpy(G)

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
                social_good_score = rate_social_good(G, decay_func)
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

    RAND = np.random.default_rng()

    avg_social_goods = []
    for i in range(20):
        # for j in range(20):
        j = 4
        social_goods = []
        print('i:', i)
        for _ in range(1000):
            inner_degrees = np.round(RAND.poisson(i, 20))
            if np.sum(inner_degrees) % 2 == 1:
                inner_degrees[np.argmin(inner_degrees)] += 1
            outer_degrees = np.round(RAND.poisson(j, 10))
            if np.sum(outer_degrees) % 2 == 1:
                outer_degrees[np.argmin(outer_degrees)] += 1
            graph, _ = cc.make_connected_community_network(inner_degrees, outer_degrees, RAND)
            social_goods.append(rate_social_good(graph))
        avg_social_good = sum(social_goods) / len(social_goods)
        avg_social_goods.append(avg_social_good)

    print(avg_social_goods)
    plt.hist(avg_social_goods)
    plt.show()
    # nx.draw(graph)
    # plt.show()

if __name__ == '__main__':
    try:
        main()
    except EOFError:
        print('\nGood-bye.')
    except KeyboardInterrupt:
        print('\nGood-bye.')
