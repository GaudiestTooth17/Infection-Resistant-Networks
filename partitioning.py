from customtypes import Communities
from itertools import takewhile
from typing import Dict, Iterable, Union, Tuple
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
from networkx.algorithms.community import girvan_newman
from fileio import old_read_network_file, get_network_name
from analyzer import make_meta_community_layout, make_meta_community_network, visualize_graph
from collections import Counter


def main():
    """
    Initial testing indicates that this is a REALLY good community detection algorithm!!
    It didn't do great on the elitist network, but perhaps with more time it could
    get better results.

    It was really crappy on Watts-Strogatz.
    """
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <network> <num labels/partitions>')
        return

    M, layout = old_read_network_file(sys.argv[1])
    if layout is None:
        raise Exception('Layout cannot be None.')
    name = get_network_name(sys.argv[1])
    n_labels = int(sys.argv[2])

    G = nx.Graph(M)
    to_remove = label_partition(G, n_labels)
    # to_remove = girvan_newman_partition(G, n_labels)  # n_labels is actually num_communities
    G.remove_edges_from(to_remove)
    # communities = tuple(nx.connected_components(G))
    # plt.hist(tuple(len(comm) for comm in communities), bins=None)
    # plt.figure()
    # plt.hist(tuple(len(comm) for comm in communities if len(comm) < 500), bins=None)
    # plt.figure()
    # visualize_graph(G, layout, f'{name} partitioned with {n_labels} labels', block=False)
    # plt.figure()
    meta_community_network, node_size, edge_width = make_meta_community_network(to_remove, G)

    def edge_width_func(G):
        return edge_width

    meta_layout = make_meta_community_layout(meta_community_network, layout)
    visualize_graph(meta_community_network, meta_layout,
                    f'{name} meta community\n{len(meta_community_network)} communities',
                    block=True, node_size=node_size, edge_width_func=edge_width_func)
    # meta_community_network.remove_edges_from(label_partition(meta_community_network, n_labels))
    # plt.figure()
    # visualize_graph(meta_community_network, meta_layout, f'{name} meta community partitioned',
    #                 node_size=node_size, block=True)
    # plt.figure()
    # comm_degrees = degree_distributions(sorted(communities, key=lambda x: -len(x))[:10], G)
    # for i, d in enumerate(comm_degrees):
    #     plt.hist(d, bins=None)
    #     plt.title('{}, len: {}'.format(i, len(d)))
    #     plt.show(block=False)
    #     plt.figure()
    # input()

    # code for mass experiments
    # start_time = time.time()
    # with Pool(10) as p:
    #     stats = p.map(run_experiment, ((M, n_labels) for _ in range(100)), 10)

    # n_modules, n_small_modules = zip(*stats)
    # print(f'Done ({time.time()-start_time}).')
    # plt.title(f'Number of Modules ({n_labels} labels)')
    # plt.hist(n_modules, bins=None)
    # plt.figure()
    # plt.title(f'Number of Small Modules ({n_labels} labels)')
    # plt.hist(n_small_modules, bins=None)
    # plt.show()


def run_experiment(args) -> Tuple[int, int]:
    M, n_labels = args
    G = nx.Graph(M)
    edges_to_remove = label_partition(G, n_labels)
    G.remove_edges_from(edges_to_remove)
    modules = tuple(nx.connected_components(G))
    return len(modules), len(tuple(filter(lambda module: len(module) < 6, modules)))


def girvan_newman_partition(G: nx.Graph, num_communities: int) -> Tuple[Tuple[int, int], ...]:
    """Return the edges to remove in order to break G into communities."""
    communities_generator: Iterable[Tuple[int, ...]] = girvan_newman(G)
    communities = tuple(takewhile(lambda comm: len(comm) <= num_communities,
                                  communities_generator))[-1]
    id_to_community = dict(enumerate(communities))
    node_to_community = {node: comm_id
                         for comm_id, community in id_to_community.items()
                         for node in community}
    edges_to_remove = tuple(set((u, v)
                                for u, v in G.edges
                                if node_to_community[u] != node_to_community[v]))
    return edges_to_remove


def label_partition(G: nx.Graph, labels: Union[int, np.ndarray]) -> Tuple[Tuple[int, int], ...]:
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

    max_steps = 1000
    for step in range(max_steps):
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


def intercommunity_edges_to_communities(G: nx.Graph,
                                        interedges: Tuple[Tuple[int, int], ...]) -> Communities:
    """
    Return a dictionary of node to the ID of the community it belongs to.

    interedges: The edges to remove in order to break up G along community lines.
    """
    H = nx.Graph(G)
    H.remove_edges_from(interedges)
    id_to_community: Dict[int, Tuple[int, ...]] = dict(enumerate(nx.connected_components(H)))
    node_to_community = {node: comm_id
                         for comm_id, comm in id_to_community.items()
                         for node in comm}
    return node_to_community


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nGood bye.')
    except EOFError:
        print('\nGood bye.')
