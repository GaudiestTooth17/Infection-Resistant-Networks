from typing import List, Dict, Tuple, Union, Set, TypeVar, Generic
from collections import namedtuple, defaultdict
import numpy as np


# color is a string and reach is a number
Agent = namedtuple('Agent', 'color reach')
NodeColors = Union[List[str], List[Tuple[int, int, int]]]
Layout = Dict[int, Tuple[float, float]]
Number = Union[int, float]
T = TypeVar('T')


class CircularList(Generic[T]):
    def __init__(self, base_list: List[T]) -> None:
        self._list = base_list

    def __getitem__(self, i: int) -> T:
        return self._list[i % len(self._list)]


class CommunityEdges:
    """
    Takes a node in brackets and returns all the outgoing edges of the community
    it belongs to.
    """
    def __init__(self, M: np.ndarray, layout: Layout,
                 sqrt_num_communities, days_to_quarantine) -> None:
        """
        sqrt_num_communities by sqrt_num_communities cells are created and nodes get placed in one of these
        :param M: Adjacency matrix
        :param layout: layout is used to partition the graph
        :param num_communities: The number of communities to split the graph into
        """
        # create communities based off location of nodes in layout
        divisions = np.linspace(-1, 1, sqrt_num_communities+1)

        def find_cell_index(x) -> int:
            for i, _ in enumerate(divisions[:-1]):
                if divisions[i] <= x <= divisions[i+1]:
                    return i
            raise Exception(f'Cannot find {x}')

        node_to_community = {node: find_cell_index(x)*sqrt_num_communities+find_cell_index(y)
                             for node, (x, y) in layout.items()}

        edges = zip(*np.where(M > 0))  # type: ignore
        community_to_outgoing_edges = defaultdict(lambda: set())
        for u, v in edges:  # type: ignore
            if node_to_community[u] != node_to_community[v]:
                community_to_outgoing_edges[node_to_community[u]].add((u, v))

        self._node_to_community: Dict[int, Tuple[int, int]] = node_to_community
        self._community_to_outgoing_edges = {community: outgoing_edges
                                             for community, outgoing_edges in community_to_outgoing_edges.items()}
        self._M = np.copy(M)
        self.c_quarantine = np.zeros(sqrt_num_communities**2)
        self.days_to_quarantine = days_to_quarantine

    def quarantine_community_by_id(self, communtity_id):
        for u, v in self._community_to_outgoing_edges[communtity_id]:
            self._M[u, v] = 0
            self._M[v, u] = 0

    def unquarantine_community_by_id(self, communtity_id):
        for u, v in self._community_to_outgoing_edges[communtity_id]:
            self._M[u, v] = 1
            self._M[v, u] = 1

    def quarantine_community(self, agents: np.ndarray) -> np.ndarray:
        """
        :param agent: indices of agent whose community will be quarantined
        """
        self.c_quarantine += (self.c_quarantine > 0)

        communities_to_unquarantine = np.where(self.c_quarantine > self.days_to_quarantine)[0]
        for c in communities_to_unquarantine:
            self.c_quarantine[c] = 0
            self.unquarantine_community_by_id(c)

        agents = np.where(agents)[0]
        communities_to_quarantine = {self._node_to_community[agent] for agent in agents}  # type: ignore
        # print('q communities:', communities_to_quarantine)

        for c in communities_to_quarantine:
            if self.c_quarantine[c] == 0:
                self.c_quarantine[c] = 1
                self.quarantine_community_by_id(c)

        return self._M

    def get_community_outgoing_edges(self, agent: int) -> Set[Tuple[int, int]]:
        """
        Returns the outgoing edges of the community that agent belongs to.
        """
        return self._community_to_outgoing_edges[self._node_to_community[agent]]
