#!/usr/bin/python3

import matplotlib.pyplot as plt
import collections
import networkx as nx
import numpy as np
import sys
from typing import Dict, List, Set, Tuple, Optional


Layout = Dict[int, Tuple[float, float]]


def read_file(fileName) -> Tuple[np.ndarray, Optional[Layout]]:
    with open(fileName, 'r') as f:
        line = f.readline()
        shape = (int(line[:-1]), int(line[:-1]))
        matrix = np.zeros(shape, dtype='uint8')

        line = f.readline()[:-1]
        i = 1
        while len(line) > 0:
            coord = line.split(' ')
            matrix[int(coord[0]), int(coord[1])] = 1
            matrix[int(coord[1]), int(coord[0])] = 1
            line = f.readline()[:-1]
            i += 1

        rest_of_lines = tuple(map(lambda s: s.split(),
                              filter(lambda s: len(s) > 1, f.readlines())))
        layout = {int(line[0]): (float(line[1]), float(line[2]))
                  for line in rest_of_lines} if len(rest_of_lines) > 0 else None
    return matrix, layout


def show_deg_dist_from_matrix(matrix: np.ndarray, title, *, color='b', display=False, save=False):
    """
    This shows a degree distribution from a matrix.

    :param matrix: The matrix.
    :param title: The title.
    :param color: The color of the degree distribution.
    :param display: Whether or not to display it.
    :param save: Whether or not to save it.
    :return: None
    """

    graph = nx.from_numpy_matrix(matrix)
    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color=color)

    plt.title(title)
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    if display:
        plt.show()
    if save:
        # plt.savefig(title[:-4] + '.png')  # This line just saves a blank pic instead of the plot.
        with open(title + '.csv', 'w') as file:
            for i in range(len(cnt)):
                file.write(f'{deg[i]},{cnt[i]}\n')
        # print(title + ' saved')
    plt.clf()


def make_node_to_degree(adj_mat) -> List[int]:
    node_to_degree = [0 for _ in range(adj_mat.shape[0])]
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if adj_mat[i][j] > 0:
                node_to_degree[i] += 1
    return node_to_degree


def show_clustering_coefficent_dist(node_to_coefficient: Dict[int, float], node_to_degree: Dict[int, int]) -> None:
    degree_to_avg_coefficient = {}
    for node, coefficient in node_to_coefficient.items():
        if node_to_degree[node] not in degree_to_avg_coefficient:
            degree_to_avg_coefficient[node_to_degree[node]] = []
        degree_to_avg_coefficient[node_to_degree[node]].append(coefficient)
    for degree, coefficients in degree_to_avg_coefficient.items():
        degree_to_avg_coefficient[degree] = sum(coefficients)/len(coefficients)

    plot_data = list(degree_to_avg_coefficient.items())
    plot_data.sort(key=lambda e: e[0])
    plt.plot(tuple(e[0] for e in plot_data), tuple(e[1] for e in plot_data))
    plt.xlabel('degree')
    plt.ylabel('average clustering coefficient')

    avg_clustering_coefficient = sum((e[1] for e in plot_data)) / len(plot_data)
    print(f'Average clustering coefficient for all nodes: {avg_clustering_coefficient}')

    plt.show()


def calc_edge_density(adj_mat) -> float:
    num_edges = 0
    for i in range(adj_mat.shape[0]):
        for j in range(i+1, adj_mat.shape[1]):
            if adj_mat[i][j] > 0:
                num_edges += 1
    density = num_edges / (adj_mat.shape[0]*(adj_mat.shape[0]-1)/2)
    return density


def get_components(graph) -> List[Set]:
    """
    returns a list of the components in graph
    :param graph: a networkx graph
    """
    return list(nx.connected_components(graph))


# shows degree distribution, degree assortativity coefficient, clustering coefficient,
# edge density
def analyze_graph(adj_matrix, name, layout) -> None:
    # edge_density = calc_edge_density(adj_mat)
    # dac = nx.degree_assortativity_coefficient(G)
    # clustering_coefficients = nx.clustering(G)
    # node_to_degree = make_node_to_degree(adj_mat)
    # components = get_components(G)

    # print(f'Edge density: {edge_density}')
    # print(f'Degree assortativity coefficient: {dac}')
    # show_clustering_coefficent_dist(clustering_coefficients, node_to_degree)
    # print(f'size of components: {[len(comp) for comp in components]}')
    show_deg_dist_from_matrix(adj_mat, name, display=True, save=True)


def visualize_graph(adj_mat: np.ndarray, layout: Optional[Layout], name='', save=False) -> None:
    G = nx.Graph(adj_mat)
    node_color = ['blue']*len(G.edges)
    if layout is None:
        nx.draw_kamada_kawai(G, node_size=50, node_color=node_color)
    else:
        nx.draw_networkx(G, pos=layout, node_size=100, with_labels=False, node_color=node_color)
    plt.title(name)
    if save:
        plt.savefig(name, dpi=300)
    else:
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <network>')
    adj_mat, layout = read_file(sys.argv[1])
    # analyze_graph(adj_mat, sys.argv[1][:-4], layout)
    visualize_graph(adj_mat, layout, sys.argv[1][:-4], True)
