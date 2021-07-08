from customtypes import Number
from typing import Callable
import numpy as np
import networkx as nx
import fileio as fio
import os
from tqdm import tqdm
import csv


def main():
    networks = ('cavemen-50-10', 'elitist-500', 'agent-generated-500',
                'annealed-agent-generated-500', 'barabasi-albert-500-3', 'cgg-500',
                'connected-comm-50-10', 'spatial-network', 'watts-strogatz-500-4-.1')
    network_paths = ['networks/'+name+'.txt' for name in networks]
    no_errors = True
    for path in network_paths:
        if not os.path.exists(path):
            print(path, 'does not exist.')
            no_errors = False
    if not no_errors:
        return
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


class DecayFunction:
    function_desc = '1/(distance^k)'

    def __init__(self, k: Number):
        self.k = k

    def __call__(self, distance: int) -> float:
        return 1/(distance**self.k)


def rate_social_good(G: nx.Graph, decay_func: Callable[[int], Number]) -> float:
    """
    Rate a network on how much social good it has.
    """
    dist_matrix = nx.floyd_warshall_numpy(G)

    def calc_score(u: int, v: int) -> float:
        return 0 if u == v else decay_func(dist_matrix[u][v])

    social_good_scores = tuple(tuple(calc_score(u, v) for v in G.nodes) for u in G.nodes)
    return np.sum(social_good_scores) / (len(G)*(len(G)-1))


if __name__ == '__main__':
    try:
        main()
    except EOFError:
        print('\nGood-bye.')
    except KeyboardInterrupt:
        print('\nGood-bye.')
