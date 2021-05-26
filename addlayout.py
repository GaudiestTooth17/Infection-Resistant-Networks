#!/usr/bin/python3

from fileio import output_network, read_network_file, get_network_name
import networkx as nx
import sys


def main(argv):
    if len(argv) < 2:
        print(f'Usage {argv[0]} <network>')
        return

    M, layout = read_network_file(argv[1])
    name = get_network_name(argv[1])
    if layout is not None:
        print('Network already has a layout.')
        return

    G = nx.Graph(M)
    layout = nx.kamada_kawai_layout(G)
    output_network(G, name+'-with-layout', layout)


if __name__ == '__main__':
    main(sys.argv)
