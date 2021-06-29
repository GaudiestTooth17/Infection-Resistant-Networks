from typing import Union, Tuple
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
from fileio import read_network_file
from analyzer import visualize_graph
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool


def main():
    """
    Initial testing indicates that this is a REALLY good community detection algorithm!!
    It didn't do great on the elitist network, but perhaps with more time it could get better results.
    It was really crappy on Watts-Strogatz.
    """
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <network> <num labels>')
        return

    M, _ = read_network_file(sys.argv[1])
    n_labels = int(sys.argv[2])

    with Pool(10) as p:
        stats = p.map(run_experiment, ((M, n_labels) for _ in range(100)), 10)
    
    n_modules, n_small_modules = zip(*stats)
    plt.title('Number of Modules')
    plt.hist(n_modules, bins=None)
    plt.figure()
    plt.title('Number of Small Modules')
    plt.hist(n_small_modules, bins=None)
    plt.show()


def run_experiment(args) -> Tuple[int, int]:
    M, n_labels = args
    G = nx.Graph(M)
    edges_to_remove = partition(G, n_labels)
    G.remove_edges_from(edges_to_remove)
    modules = tuple(nx.connected_components(G))
    return len(modules), len(tuple(filter(lambda module: len(module) < 6, modules)))


def girvan_newman_partition(G, communities) -> nx.Graph:
    G = nx.Graph(G)

    edges_to_keep = set()
    for u, v in G.edges:
        for partition in communities:
            if u in partition and v in partition:
                edges_to_keep.add((u, v))
    edges_to_remove = set(G.edges) - edges_to_keep
    G.remove_edges_from(edges_to_remove)
    return G


def partition(G: nx.Graph, labels: Union[int, np.ndarray]) -> Tuple[Tuple[int, int], ...]:
    """
    Return the edges to remove in order to break G into communities.

    G: Network to partition.
    labels: Either the number of classes of labels to randomly assign to vertices,
            or a premade array of labels to assign to the vertices.
    """
    if isinstance(labels, int):
        available_labels = tuple(range(labels))
        rand = np.random.default_rng()
        node_to_label = rand.choice(available_labels, size=len(G))
    else:
        node_to_label = labels

    for _ in range(100):
        # do label propagation
        new_labels = np.zeros(len(G), dtype=np.int64)
        for node in G:
            label_counts = Counter([node_to_label[neighbor] for neighbor in G[node]]
                                   + [node_to_label[node]])
            most_popular = max(label_counts.items(), key=lambda x: x[1])[0]
            new_labels[node] = most_popular

        # early exit condition
        if (node_to_label == new_labels).all():
            break
        node_to_label = new_labels

    edges_to_remove = filter(lambda edge: node_to_label[edge[0]] != node_to_label[edge[1]], G.edges)
    return tuple(edges_to_remove)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nGood bye.')
    except EOFError:
        print('\nGood bye.')
