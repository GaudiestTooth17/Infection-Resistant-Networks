import sys
import fileio as fio
import networkx as nx
from analysis import (analyze_network, visualize_network, all_same,
                      make_meta_community_layout, make_meta_community_network)


def analyze_network_entry_point():
    """Analyze the network given by the command line argument."""
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <network>')
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


if __name__ == '__main__':
    try:
        analyze_network_entry_point()
    except KeyboardInterrupt:
        print('\nGood bye.')
    except EOFError:
        print('\nGood bye.')
