import sys
from typing import List
import numpy as np
import networkx as nx
import itertools as it
sys.path.append('')
from network import Network


def flatten(sequence):
    return tuple(it.chain(*sequence))


def make_affiliation_network(group_to_membership_percentage: List[float],
                             N: int, rng) -> Network:
    """
    Nodes in association networks are connected if they belong to at least one common group.
    To generate an association network, first a bipartite network is formed with one set of
    nodes being the groups and the other the agents. The next step is to add an edge between
    all agents that share membership in at least one group. Finally, the group nodes and any
    edges attached to them are removed.

    group_to_membership_percentage: each index is associated with a group's ID and each value
                                    is what percentage of agents belong to that group
    N: The number of agents in the network
    rng: an np.random.default_rng instance
    """
    agents = tuple(range(N))
    group_memberships = [rng.choice(agents, size=int(np.round(N*perc)), replace=False)
                         for perc in group_to_membership_percentage]
    edges = flatten(tuple(it.combinations(membership, 2)) for membership in group_memberships)
    # If the graph is construct solely from the edge list, some nodes might be left out.
    # So, construct an empty graph and then add the edges
    G: nx.Graph = nx.empty_graph(N)
    G.add_edges_from(edges)
    return Network(G)
