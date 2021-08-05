import sys
import fileio as fio
import networkx as nx
from analysis import (analyze_network, visualize_network, all_same,
                      make_meta_community_layout, make_meta_community_network)
import numpy as np
import matplotlib.pyplot as plt
import time
import socialgood


def analyze_network_entry_point():
    """Analyze the network given by the command line argument."""
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <network>')
        return

    net = fio.read_network(sys.argv[1])
    G = nx.Graph(net.G)
    name = fio.get_network_name(sys.argv[1])
    analyze_network(G, name)
    visualize_network(G, net.layout, name, edge_width_func=all_same, block=False)
    intercommunity_edges = tuple((u, v) for u, v in G.edges
                                 if net.communities[u] != net.communities[v])
    G.remove_edges_from(intercommunity_edges)
    visualize_network(G, net.layout, 'Partitioned '+name, edge_width_func=all_same, block=False)
    meta_G, meta_ns, meta_ew = make_meta_community_network(intercommunity_edges, G)
    meta_layout = make_meta_community_layout(meta_G, net.layout)
    visualize_network(meta_G, meta_layout, f'{name} Meta Communities',
                      edge_width_func=lambda G: meta_ew, node_size=meta_ns, block=False)
    input('Press <enter> to exit.')


def visualize_communicability():
    if len(sys.argv) < 2:
        print(f'Usage {sys.argv[1]} <network>')
        return

    net = fio.read_network(sys.argv[1])
    name = fio.get_network_name(sys.argv[1])
    communicability = nx.communicability(net.G)
    scores = np.array([sum(communicability[u].values()) for u in communicability.keys()])
    plt.title(f'{name}\nCommunicability')
    plt.hist(scores)  # type: ignore
    plt.show(block=False)
    print(f'Network score: {np.sum(scores)}')
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    node_size = 300 * scores
    visualize_network(net.G, net.layout, name, node_size=node_size)


def show_edit_distance():
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <network 0> <network 1>')
        return

    start_time = time.time()
    net0 = fio.read_network(sys.argv[1])
    net1 = fio.read_network(sys.argv[2])
    distance = nx.graph_edit_distance(net0.G, net1.G)
    name0 = fio.get_network_name(sys.argv[1])
    name1 = fio.get_network_name(sys.argv[2])
    print(f'Distance between {name0} and {name1}: {distance} ({time.time()-start_time} s)')


def show_social_good():
    if len(sys.argv) < 2:
        print(f'Usage {sys.argv[1]} <network>')
        return

    net = fio.read_network(sys.argv[1])
    for k in np.linspace(.5, 1.5, 5):
        sg_score = socialgood.rate_social_good(net, socialgood.DecayFunction(k))
        print(f'{k:<6.3f}: {sg_score}')


if __name__ == '__main__':
    try:
        show_social_good()
    except KeyboardInterrupt:
        print('\nGood bye.')
    except EOFError:
        print('\nGood bye.')
