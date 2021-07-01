#!/usr/bin/python3

from fileio import old_output_network, old_read_network_file, get_network_name
import networkx as nx
import sys
import time


def main(argv):
    if len(argv) < 2:
        print(f'Usage {argv[0]} <network>')
        return

    M, layout = old_read_network_file(argv[1])
    name = get_network_name(argv[1])

    G = nx.Graph(M)
    start_time = time.time()
    # layout = nx.kamada_kawai_layout(G)
    layout = nx.spring_layout(G, iterations=50, pos=layout)
    old_output_network(G, name+'-with-layout', layout)
    print(f'Done ({time.time()-start_time}).')


if __name__ == '__main__':
    main(sys.argv)
