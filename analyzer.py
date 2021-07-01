#!/usr/bin/python3

import matplotlib.pyplot as plt
import collections
import networkx as nx
import numpy as np
from tqdm import tqdm
import sys
from itertools import takewhile
from typing import Callable, Counter, Dict, Iterable, List, Sequence, Set, Optional, Tuple, Union
from customtypes import Layout, Number, CircularList
from fileio import old_read_network_file, get_network_name
RAND = np.random.default_rng()


COLORS = CircularList(['blue', 'green', 'lightcoral', 'chocolate', 'darkred',
                       'navy', 'darkorange', 'springgreen', 'darkcyan', 'indigo',
                       'slategrey', 'darkgreen', 'crimson', 'magenta', 'darkviolet',
                       'palegreen', 'goldenrod', 'darkolivegreen'])


def main(argv: List[str]):
    if len(argv) < 2:
        print(f'Usage: {argv[0]} <network>')
    M, _ = old_read_network_file(argv[1])
    name = get_network_name(argv[1])
    analyze_network(nx.Graph(M), name)
    # visualize_graph(nx.Graph(M), layout, name, edge_width_func=all_same, save=False)
    # visualize_eigen_communities(nx.Graph(M), layout, name)
    # visualize_girvan_newman_communities(nx.Graph(M), layout, name)
    # plot_edge_betweeness_centralities(nx.Graph(M), name)


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
        plt.show(block=False)
    if save:
        # plt.savefig(title[:-4] + '.png')  # This line just saves a blank pic instead of the plot.
        with open(title + '.csv', 'w') as file:
            for i in range(len(cnt)):
                file.write(f'{deg[i]},{cnt[i]}\n')
        # print(title + ' saved')
    plt.figure()


def make_node_to_degree(M) -> List[int]:
    node_to_degree = [0 for _ in range(M.shape[0])]
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i][j] > 0:
                node_to_degree[i] += 1
    return node_to_degree


def show_clustering_coefficent_dist(node_to_coefficient: Dict[int, float],
                                    node_to_degree: Dict[int, int]) -> None:
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

    plt.show(block=False)
    plt.figure()


def calc_edge_density(M) -> float:
    num_edges = 0
    for i in range(M.shape[0]):
        for j in range(i+1, M.shape[1]):
            if M[i][j] > 0:
                num_edges += 1
    density = num_edges / (M.shape[0]*(M.shape[0]-1)/2)
    return density


def get_components(graph) -> Tuple[Set, ...]:
    """
    returns a list of the components in graph
    :param graph: a networkx graph
    """
    return tuple(nx.connected_components(graph))


# shows degree distribution, degree assortativity coefficient, clustering coefficient,
# edge density
def analyze_network(G: nx.Graph, name) -> None:
    M = nx.to_numpy_array(G)
    dac = nx.degree_assortativity_coefficient(G)
    clustering_coefficients = nx.clustering(G)
    node_to_degree = make_node_to_degree(M)

    edge_density = calc_edge_density(M)
    print(f'Edge density: {edge_density}')
    if nx.is_connected(G):
        diameter = nx.diameter(G)
    else:
        largest_component = max(nx.connected_components(G), key=len)
        largest_subgraph = G.subgraph(largest_component)
        diameter = nx.diameter(largest_subgraph)
    print(f'Diameter: {diameter}')
    print(f'Degree assortativity coefficient: {dac}')
    show_clustering_coefficent_dist(clustering_coefficients,  # type: ignore
                                    dict(enumerate(node_to_degree)))
    # components = get_components(G)
    # print(f'size of components: {[len(comp) for comp in components]}')
    show_deg_dist_from_matrix(M, name, display=False, save=True)
    input('Press <enter> to continue.')


def all_same(G: nx.Graph) -> List[float]:
    """
    All the edges are the same width.
    """
    return [1.0 for _ in G.edges]


def betw_centrality(G: nx.Graph) -> List[float]:
    """
    Edge width depends on betweeness centrality.
    (higher centrality -> more width)
    """
    return [5*betweenness for betweenness in nx.edge_betweenness_centrality(G).values()]


def common_neigh(G: nx.Graph) -> List[float]:
    """
    Edge width depends on how many common neighbors the two end points have.
    (more in common -> more width).
    """
    return [.1+2*calc_prop_common_neighbors(G, u, v) for u, v in G.edges]


def rw_centrality(G: nx.Graph) -> List[float]:
    # This doesn't do a good job of distinguishing between edges
    centralities = random_walk_centrality(G, 10000)
    width = [1500 * centrality for centrality in centralities.values()]
    plt.hist(width, bins=None)
    plt.show(block=False)
    plt.figure()
    return width


def visualize_graph(G: nx.Graph, layout: Optional[Layout], name='', save=False,
                    edge_width_func: Callable[[nx.Graph], Sequence[float]] = all_same,
                    block=True, node_size: Union[int, Sequence[int]] = 50,
                    node_color: Optional[Sequence[str]] = None) -> None:
    comps = tuple(nx.connected_components(G))
    if node_color is None:
        node_color = colors_from_communities(comps)
    edge_width = edge_width_func(G)
    plt.title(f'{name}\n{len(comps)} Components')
    # node_size = np.array(tuple(nx.betweenness_centrality(G).values()))
    # node_size = np.array(tuple(nx.eigenvector_centrality_numpy(G).values()))
    if layout is None:
        nx.draw_kamada_kawai(G, node_size=node_size, node_color=node_color, width=edge_width)
    else:
        nx.draw_networkx(G, pos=layout, node_size=node_size, node_color=node_color,
                         width=edge_width, with_labels=False)
    if save:
        plt.savefig(f'vis-{name}.png', dpi=300, format='png')
    else:
        plt.show(block=block)


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


def colors_from_communities(communities: Sequence[Sequence[int]]) -> List[str]:
    colors = [(COLORS[i], vertex)
              for i, community in enumerate(communities)
              for vertex in community]
    colors.sort(key=lambda x: x[1])
    return [x[0] for x in colors]  # type: ignore


def make_partitioner(partitioning_vector: np.ndarray) -> Callable[[float], List[str]]:
    """
    Return a partitioning closure for the graph whose Laplacian's fiedler vector is provided.

    The closure iterates over partitioning_vector looking for gaps larger or equal to the cutoff
    it is passed. When one of these gaps is found, the new community begins accumulating members.
    A list of color strings for use with NetworkX drawing functions is returned.
    """
    node_to_value = sorted(enumerate(partitioning_vector), key=lambda x: x[1])  # type: ignore

    def partitioner(community_cutoff: float) -> List[str]:
        """
        This doesn't contain much actual partitioning logic. It just assigns colors to nodes
        based on the cutoff value for determining communities.
        """
        community = 0
        community_colors = ['black'] * len(node_to_value)
        node = 0
        # assign every node but the last one to a community
        while node < len(node_to_value)-1:
            if np.abs(node_to_value[node][1] - node_to_value[node+1][1]) >= community_cutoff:
                community += 1
            community_colors[node] = COLORS[community]  # type: ignore
            node += 1
        # assign the last node to a community

        if np.abs(node_to_value[node-1][1]-node_to_value[node][1]) < community_cutoff:
            community_colors[node] = COLORS[community]  # type: ignore
        else:
            community_colors[node] = COLORS[community+1]  # type: ignore
        return community_colors

    return partitioner


def plot_edge_betweeness_centralities(G: nx.Graph, name: str) -> None:
    centralities = nx.edge_betweenness_centrality(G)
    plt.title(f'{name} Edge Centralities')
    plt.hist(centralities.values(), bins=None)  # type: ignore
    plt.savefig(name+' edge betweeness.png', format='png')


def normalize(xs: Iterable[Number]) -> np.ndarray:
    max_x = max(xs)
    return np.array([x/max_x for x in xs])


def calc_prop_common_neighbors(G: nx.Graph, u: int, v: int) -> float:
    u_neighbors = set(nx.neighbors(G, u))
    v_neighbors = set(nx.neighbors(G, v))
    n_common_neighbors = len(u_neighbors.intersection(v_neighbors))
    return n_common_neighbors / len(u_neighbors)


def random_walk_centrality(G: nx.Graph, num_paths: int) -> Dict[Tuple[int, int], float]:
    """
    Sample num_paths paths to calculate the random walk centrality of the edges in G.
    """
    rand = RAND
    edge_to_times_crossed = collections.defaultdict(lambda: 0)
    edges_crossed = 0
    for _ in tqdm(range(num_paths)):
        s, t = rand.choice(G), rand.choice(G)
        current = s
        while current != t:
            next = rand.choice(G[current])
            edge_to_times_crossed[(current, next)] += 1
            edges_crossed += 1
            current = next

    return {edge: frequency/edges_crossed for edge, frequency in edge_to_times_crossed.items()}


def make_meta_community_network(edges_removed: Tuple[Tuple[int, int], ...],
                                partitioned_G: nx.Graph)\
                                    -> Tuple[nx.Graph, Sequence[int], Sequence[float]]:
    """
    Make a network where each node represents a partition in the original network
    and each edge represents at least one edge going between the two partitions.

    The nodes in the meta community network have the nodes in the communities
    they represent associated with them as attributes with the tab 'communities'.
    The edges have a weight associated with them proportional to the number of
    edges that crossed between the partitions in the original network.

    Returns the meta community network, suggested node_size, suggested edge_width
    """
    communities = dict(enumerate(nx.connected_components(partitioned_G)))
    node_to_community = {}

    def find_community(node):
        if node in node_to_community:
            return node_to_community[node]
        for community_id, nodes in communities.items():
            if node in nodes:
                node_to_community[node] = community_id
                return community_id
        raise Exception(f'Cannot find {node}')

    community_network: nx.Graph = nx.empty_graph(len(communities))
    edge_to_weight = Counter((find_community(u), find_community(v)) for u, v in edges_removed)
    community_network.add_edges_from(edge_to_weight.keys())
    nx.set_node_attributes(community_network, communities, 'communities')
    nx.set_edge_attributes(community_network, edge_to_weight, 'weight')

    node_size = np.array([len(community) for community in communities.values()])
    node_size = node_size / np.sum(node_size) * 1000
    edge_width = np.array(tuple(edge_to_weight.values()))
    edge_width = edge_width / np.sum(edge_width) * 20
    return community_network, node_size, edge_width


def degree_distributions(components: Iterable[Sequence[int]],
                         G: nx.Graph) -> Sequence[Sequence[int]]:
    """Return the degree distributions for each of the components in G."""
    comm_degrees = []
    for comm in components:
        degrees = [G.degree[n] for n in comm]
        comm_degrees.append(degrees)
    return comm_degrees


def make_meta_community_layout(meta_G: nx.Graph, original_layout: Layout) -> Layout:
    """Make a layout for a meta community network based on the original network's layout."""
    communities = nx.get_node_attributes(meta_G, 'communities')
    layout = {node: np.average([original_layout[n] for n in communities[node]], axis=0)
              for node in meta_G}
    return layout


if __name__ == '__main__':
    try:
        main(sys.argv)
    except EOFError:
        print('\nGood-bye.')
