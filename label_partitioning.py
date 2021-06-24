import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
from fileio import read_network_file
from analyzer import COLORS, visualize_graph
from collections import Counter


def main():
    """
    Initial testing indicates that this is a REALLY good community detection algorithm!!
    It didn't do great on the elitist network, but perhaps with more time it could get better results.
    It was really crappy on Watts-Strogatz.
    """
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <network> <num labels>')
        return

    M, layout = read_network_file(sys.argv[1])
    G = nx.Graph(M)

    try:
        labels = COLORS[:int(sys.argv[2])]
    except ValueError:
        print('provide an integer number of labels.')
        exit(1)

    rand = np.random.default_rng()
    # randomly assign labels
    nx.set_node_attributes(G, {node: rand.choice(labels) for node in G}, 'label')
    node_to_label = nx.get_node_attributes(G, 'label')
    nx.draw_networkx(G, with_labels=False, node_size=50, pos=layout,
                     node_color=node_to_label.values())
    plt.pause(1)

    for step in range(100):
        node_to_label = nx.get_node_attributes(G, 'label')
        new_labels = {}
        for node in G:
            label_counts = Counter([node_to_label[neighbor] for neighbor in G[node]] + [node_to_label[node]])
            most_popular = max(label_counts.items(), key=lambda x: x[1])[0]
            new_labels[node] = most_popular

        nx.set_node_attributes(G, new_labels, 'label')
        plt.clf()
        nx.draw_networkx(G, with_labels=False, node_size=50, pos=layout,
                         node_color=new_labels.values())
        plt.pause(.2)  # type: ignore
        if node_to_label == new_labels:
            print(f'Done. {step} steps.')
            break

    edges_to_remove = filter(lambda edge: node_to_label[edge[0]] != node_to_label[edge[1]], G.edges)
    G.remove_edges_from(edges_to_remove)
    plt.clf()
    visualize_graph(nx.to_numpy_array(G), layout, 'Final Version')


if __name__ == '__main__':
    main()
