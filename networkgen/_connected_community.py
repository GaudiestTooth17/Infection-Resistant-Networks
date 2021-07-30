from customtypes import Communities, Layout
import numpy as np
from fileio import write_network
from analysis import visualize_network
from typing import Optional
import networkx as nx
from network import Network
RAND = np.random.default_rng()


def connected_community_entry_point():
    N_comm = 10
    num_communities = 50
    name = f'connected-comm-{num_communities}-{N_comm}'

    inner_degrees = np.round(RAND.poisson(10, N_comm))
    if np.sum(inner_degrees) % 2 == 1:
        inner_degrees[np.argmin(inner_degrees)] += 1
    outer_degrees = np.round(RAND.poisson(4, num_communities))
    if np.sum(outer_degrees) % 2 == 1:
        outer_degrees[np.argmin(outer_degrees)] += 1

    net = make_connected_community_network(inner_degrees, outer_degrees)
    if net is None:
        print('Could not generate network.')
        exit(1)
    net

    layout: Layout = nx.spring_layout(net.G, iterations=200)
    visualize_network(net.G, layout, name, block=False)
    # analyze_network(net.G, name)
    if input('Save? ').lower() == 'y':
        write_network(net.G, name, layout, net.communities)


def make_connected_community_network(inner_degrees: np.ndarray,
                                     outer_degrees: np.ndarray,
                                     rand=RAND,
                                     max_tries: int = 100,
                                     allow_disconnected: bool = True)\
        -> Optional[Network]:
    """
    Create a random network divided into randomly generated communities connected to each other.

    The number of vertices in each community is determined by the length of inner_degrees.
    Similarly, the number of communities is determined by the length of outer_degrees.

    inner_degrees: the inner degree of each of the vertices in the communities.
    outer_degrees: How many outgoing edges each of the communities has.
    return: A network and dictionary of node to community or None on time out.
    """
    node_to_community = {}
    num_communities = len(outer_degrees)
    community_size = len(inner_degrees)
    N = community_size * num_communities
    for _ in range(max_tries):
        M = np.zeros((N, N), dtype=np.uint8)
        node_to_community: Communities = {}

        for community_id in range(num_communities):
            c_offset = community_id * community_size
            comm_M = make_configuration_network(inner_degrees, rand)
            for n1 in range(community_size):
                for n2 in range(community_size):
                    M[n1 + c_offset, n2 + c_offset] = comm_M[n1, n2]
                    node_to_community[n1+c_offset] = community_id
                    node_to_community[n2+c_offset] = community_id

        outer_m = make_configuration_network(outer_degrees, rand)
        for c1 in range(num_communities):
            for c2 in range(c1, num_communities):
                if outer_m[c1, c2] > 0:
                    for _ in range(int(outer_m[c1, c2])):
                        n1 = c1 * community_size + rand.integers(community_size)
                        n2 = c2 * community_size + rand.integers(community_size)
                        M[n1, n2] = 1
                        M[n2, n1] = 1

        G = nx.Graph(M)
        if allow_disconnected or nx.is_connected(G):
            G.remove_edges_from(nx.selfloop_edges(G))
            return Network(G, communities=node_to_community)

    return None


def make_configuration_network(degree_distribution: np.ndarray, rand) -> np.ndarray:
    """Create a random network with the provided degree distribution."""
    degree_distribution = np.copy(degree_distribution)

    N = degree_distribution.shape[0]
    M = np.zeros((N, N), dtype=np.uint8)

    while np.sum(degree_distribution > 0):
        a = rand.choice(np.where(degree_distribution > 0)[0])
        degree_distribution[a] -= 1
        b = rand.choice(np.where(degree_distribution > 0)[0])
        degree_distribution[b] -= 1

        M[a, b] += 1
        M[b, a] += 1

    return M
