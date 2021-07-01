"""Converts from the old network format to the new one."""

import sys
from partitioning import label_partition, intercommunity_edges_to_communities
from fileio import old_read_network_file, get_network_name, write_network, read_network
from analyzer import (visualize_graph, COLORS, make_meta_community_network,
                      make_meta_community_layout)
import networkx as nx
import matplotlib.pyplot as plt


"""
networks to work on
annealed-short-diameter: Edit this manually.
barabasi-albert-500-3: This one might not be salvagable
example-cgg: make sure this has the correct number of communities.
cavemen-50-10: It doesn't look like GN worked. You should double check.
cgg-520: I should just update the function to output communities.
connected-community-100-10: see cgg-520
evolved: I should just get rid of this entirely.
spatial-network: I just need to verify the number of communities.
watts-strogatz-500-4-.1: It looks like this was partitioned into 10
"""


def main():
    if len(sys.argv) < 3:
        print(f'Usage {sys.argv[0]} <network> <num communities>')
        return

    M, layout = old_read_network_file(sys.argv[1])
    if layout is None:
        print('layout is None. Exiting.')
        return

    G = nx.Graph(M)
    interedges = label_partition(G, int(sys.argv[2]))
    communities = intercommunity_edges_to_communities(G, interedges)

    new_name = get_network_name(sys.argv[1])+'-new-format'
    write_network(G, new_name, layout, communities)

    G, layout, communities = read_network(new_name+'.txt')
    if layout is None:
        print(f'layout for {new_name} was not saved correctly. Exiting')
        return
    if communities is None:
        print('communities is None. Exiting.')
        return

    interedges = tuple(set((u, v) for u, v in G.edges
                           if communities[u] != communities[v]))
    partitioned = nx.Graph(G)
    partitioned.remove_edges_from(interedges)
    meta_network, meta_ns, meta_ew = make_meta_community_network(interedges, partitioned)
    meta_layout = make_meta_community_layout(meta_network, layout)

    node_color = [COLORS[i] for i in communities.values()]
    visualize_graph(G, layout, new_name, node_color=node_color, block=False)  # type: ignore
    plt.figure()
    visualize_graph(meta_network, meta_layout, new_name+'meta community',
                    node_size=meta_ns, edge_width_func=lambda G: meta_ew)


if __name__ == '__main__':
    main()
