#!/usr/bin/python3

import matplotlib.pyplot as plt
import collections
import networkx as nx
import numpy as np
import sys
from itertools import takewhile
from typing import Callable, Dict, Iterable, List, Set, Tuple, Optional, Union
from circularlist import CircularList


Layout = Dict[int, Tuple[float, float]]
Number = Union[int, float]
COLORS = CircularList(['blue', 'green', 'lightcoral', 'chocolate', 'darkred',
                       'navy', 'darkorange', 'springgreen', 'darkcyan', 'indigo',
                       'slategrey', 'darkgreen', 'crimson', 'magenta', 'darkviolet',
                       'palegreen', 'goldenrod', 'darkolivegreen'])


def main(argv: List[str]):
    if len(argv) < 2:
        print(f'Usage: {argv[0]} <network>')
    M, layout = read_file(argv[1])
    # analyze_graph(M, argv[1][:-4], layout)
    # visualize_graph(M, layout, argv[1][:-4], show_edge_betweeness=True)
    # visualize_eigen_communities(nx.Graph(M), layout, argv[1][:-4])
    visualize_girvan_newman_communities(nx.Graph(M), layout, argv[1][:-4])


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


def show_deg_dist_from_matrix(M: np.ndarray, title, *, color='b', display=False, save=False):
    """
    This shows a degree distribution from a matrix.

    :param matrix: The matrix.
    :param title: The title.
    :param color: The color of the degree distribution.
    :param display: Whether or not to display it.
    :param save: Whether or not to save it.
    :return: None
    """

    graph = nx.from_numpy_matrix(M)
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


def make_node_to_degree(M) -> List[int]:
    node_to_degree = [0 for _ in range(M.shape[0])]
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i][j] > 0:
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


def calc_edge_density(M) -> float:
    num_edges = 0
    for i in range(M.shape[0]):
        for j in range(i+1, M.shape[1]):
            if M[i][j] > 0:
                num_edges += 1
    density = num_edges / (M.shape[0]*(M.shape[0]-1)/2)
    return density


def get_components(graph) -> List[Set]:
    """
    returns a list of the components in graph
    :param graph: a networkx graph
    """
    return list(nx.connected_components(graph))


# shows degree distribution, degree assortativity coefficient, clustering coefficient,
# edge density
def analyze_graph(M, name, layout) -> None:
    # dac = nx.degree_assortativity_coefficient(G)
    # clustering_coefficients = nx.clustering(G)
    # node_to_degree = make_node_to_degree(M)
    # components = get_components(G)

    edge_density = calc_edge_density(M)
    print(f'Edge density: {edge_density}')
    diameter = nx.diameter(nx.Graph(M))
    print(f'Diameter: {diameter}')
    # print(f'Degree assortativity coefficient: {dac}')
    # show_clustering_coefficent_dist(clustering_coefficients, node_to_degree)
    # print(f'size of components: {[len(comp) for comp in components]}')
    show_deg_dist_from_matrix(M, name, display=True, save=True)


def visualize_graph(M: np.ndarray, layout: Optional[Layout], name='',
                    save=False, show_edge_betweeness=False) -> None:
    G = nx.Graph(M)
    node_color = ['blue']*len(G.edges)
    edge_width = 5*normalize(nx.edge_betweenness_centrality(G).values()) if show_edge_betweeness else 1
    if layout is None:
        nx.draw_kamada_kawai(G, node_size=100, node_color=node_color, width=edge_width)
    else:
        nx.draw_networkx(G, pos=layout, node_size=100, node_color=node_color, width=edge_width)
    plt.title(name)
    if save:
        plt.savefig(name, dpi=300)
    else:
        plt.show()


def visualize_eigen_communities(G: nx.Graph, layout: Optional[Layout] = None, name='') -> None:
    """
    Helps visualize communities created with the fiedler partitioning algorithm. The user enters
    different values to act as cutoffs when determining how to partition the network and the
    resulting partitions are plotted.
    """
    L = nx.linalg.laplacian_matrix(G).toarray()  # type: ignore
    eigvs, eigenvectors = np.linalg.eigh(L)
    plt.title('Eigenvalues')
    plt.plot(eigvs)
    plt.show(block=False)
    plt.figure()
    partitioning_vector = eigenvectors[:, 1]
    plt.title('Partitioning Vector')
    plt.plot(sorted(partitioning_vector))
    plt.show(block=False)

    if layout is None:
        layout = nx.kamada_kawai_layout(G)

    partition_network = make_partitioner(partitioning_vector)
    while True:
        comm_cutoff = float(input('Max diff for communities: '))
        community_colors = partition_network(comm_cutoff)
        plt.figure()
        nx.draw_networkx(G, layout, node_color=community_colors, with_labels=False, node_size=100)
        plt.title(f'{name} cutoff = {comm_cutoff}\n{len(set(community_colors))} communities')
        plt.show(block=False)


def visualize_girvan_newman_communities(G: nx.Graph, layout: Optional[Layout] = None,
                                        name='', max_communities=10) -> None:
    """
    Show a visualization of the Girvan-Newman Method for network G.

    This function displays a plot of G slowly losing its edges. When new communities are formed,
    they are given a new color.
    """
    communities_generator = nx.algorithms.community.centrality.girvan_newman(G)  # type: ignore
    for communities in takewhile(lambda comms: len(comms) <= max_communities,
                                 communities_generator):
        community_colors = colors_from_communities(communities)
        plt.figure()
        nx.draw_networkx(G, layout, node_color=community_colors, with_labels=False, node_size=100)
        plt.title(f'{name}\n{len(communities)} communities')
        plt.show(block=False)

    while input('Type "continue" to continue: ') != 'continue':
        pass


def colors_from_communities(communities: Tuple[List[int]]) -> List[str]:
    colors = [(COLORS[i], vertex)
              for i, community in enumerate(communities)
              for vertex in community]
    colors.sort(key=lambda x: x[1])
    return [x[0] for x in colors]


def make_partitioner(partitioning_vector: np.ndarray) -> Callable[[float], List[str]]:
    """
    Return a partitioning closure for the graph whose Laplacian's fiedler vector is provided.

    The closure iterates over partitioning_vector looking for gaps larger or equal to the cutoff
    it is passed. When one of these gaps is found, the new community begins accumulating members.
    A list of color strings for use with NetworkX drawing functions is returned.
    """
    node_to_value = sorted(enumerate(partitioning_vector), key=lambda x: x[1])  # type: ignore

    def partitioner(community_cutoff: float) -> List[str]:
        community = 0
        community_colors = ['black'] * len(node_to_value)
        node = 0
        # assign every node but the last one to a community
        while node < len(node_to_value)-1:
            if np.abs(node_to_value[node][1] - node_to_value[node+1][1]) >= community_cutoff:
                community += 1
            community_colors[node] = COLORS[community]
            node += 1
        # assign the last node to a community
        community_colors[node] = COLORS[community]\
            if np.abs(node_to_value[node-1][1]-node_to_value[node][1]) < community_cutoff\
            else COLORS[community+1]
        return community_colors

    return partitioner


def normalize(xs: Iterable[Number]) -> np.ndarray:
    max_x = max(xs)
    return np.array([x/max_x for x in xs])


if __name__ == '__main__':
    try:
        main(sys.argv)
    except EOFError:
        print('\nGood-bye.')
