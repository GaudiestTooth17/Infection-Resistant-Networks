from typing import Tuple
import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman
import numpy as np
import matplotlib.pyplot as plt
import sys
from fileio import read_network_file
from analyzer import visualize_graph
from collections import Counter


def main():
    """
    Initial testing indicates that this is a REALLY good community detection algorithm!!
    It didn't do great on the elitist network, but perhaps with more time it could get better results.
    It was really crappy on Watts-Strogatz.
    """
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <network>')
        return

    M, layout = read_network_file(sys.argv[1])
    label_partitioned = nx.Graph(M)

    n_labels = 3
    edges_to_remove = partition(label_partitioned, n_labels)
    label_partitioned.remove_edges_from(edges_to_remove)
    visualize_graph(label_partitioned, layout,
                    f'Label Partitioned\n{n_labels} Labels',
                    block=True)

    G = nx.Graph(M)
    print('Running GN')
    for communities in girvan_newman(G):
        if len(communities) > 9:
            H = girvan_newman_partition(G, communities)
            plt.figure()
            visualize_graph(H, layout, 'Girvan Newman', block=False)
        print(f'Finished iteration {len(communities)}')
        if len(communities) > 20:
            break

    input('Done')


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


def partition(G: nx.Graph, num_labels: int) -> Tuple[Tuple[int, int], ...]:
    """Return the edges to remove in order to break G into communities."""
    labels = tuple(range(num_labels))
    rand = np.random.default_rng()
    node_to_label = rand.choice(labels, size=len(G))

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
