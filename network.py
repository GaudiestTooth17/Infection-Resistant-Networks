from typing import Callable, Union, Optional, Collection, Tuple, Iterable
from customtypes import Communities, Layout
import networkx as nx
import retworkx as rx
import numpy as np
from partitioning import fluidc_partition, intercommunity_edges_to_communities


class Network:
    def __init__(self, data: Union[nx.Graph, np.ndarray],
                 intercommunity_edges: Optional[Collection[Tuple[int, int]]] = None,
                 communities: Optional[Communities] = None,
                 community_size: int = 25,
                 layout: Union[Layout, Callable[[nx.Graph], Layout]] = nx.kamada_kawai_layout):
        """
        Holds both the NetworkX and NumPy representation of a network. It starts
        off with just one and lazily creates the other representation.

        Caveats:
        Do not mutate; changes in one data structure are not reflected in the other.
        Does not support selfloops or multiedges.
        """
        if isinstance(data, nx.Graph):
            # make sure that nodes are identified by integers
            if not isinstance(next(iter(data.nodes)), int):
                data = nx.relabel_nodes(data, {old: new for new, old in enumerate(data.nodes)})
            self._G: nx.Graph = data  # type: ignore
            self._M = None  # type: ignore
        else:
            self._G = None  # type: ignore
            self._M: np.ndarray = data
        self._intercommunity_edges = intercommunity_edges
        self._communities = communities
        self._community_size = community_size
        self._layout = layout
        self._edge_density = None
        self._R = None
        self._dm = None
        self._edm = None  # Edge distance matrix (distance to attached edges is 1)

    @property
    def G(self) -> nx.Graph:
        if self._G is None:
            self._G = nx.Graph(self._M)
        return self._G

    @property
    def M(self) -> np.ndarray:
        if self._M is None:
            self._M = nx.to_numpy_array(self._G)
        return self._M

    @property
    def R(self):
        """Return a retworkx PyGraph"""
        if self._R is None:
            self._R = rx.networkx_converter(self.G)
        return self._R

    @property
    def N(self) -> int:
        return self.__len__()

    @property
    def E(self) -> int:
        if self._G is not None:
            return len(self._G.edges)
        return np.sum(self._M > 0) // 2

    @property
    def edge_density(self) -> float:
        if self._edge_density is None:
            self._edge_density = self.E / ((self.N**2 - self.N)//2)
        return self._edge_density

    @property
    def edges(self) -> Iterable[Tuple[int, int]]:
        return self.G.edges

    @property
    def intercommunity_edges(self):
        if self._intercommunity_edges is None:
            if self._communities is None:
                self._intercommunity_edges = fluidc_partition(self.G, self.N//self._community_size)
            else:
                self._intercommunity_edges = tuple((u, v) for u, v in self.edges
                                                   if self._communities[u] != self._communities[v])
        return self._intercommunity_edges

    @property
    def communities(self):
        if self._communities is None:
            self._communities = intercommunity_edges_to_communities(self.G,
                                                                    self.intercommunity_edges)
        return self._communities

    @property
    def layout(self) -> Layout:
        if callable(self._layout):
            self._layout = self._layout(self.G)
        return self._layout

    @property
    def dm(self):
        if self._dm is None:
            m: np.ndarray = rx.distance_matrix(self.R).copy()  # type: ignore
            m[m == 0] = np.inf
            for u in range(len(m)):
                m[u, u] = 0
            self._dm = m
        return self._dm

    @property
    def edm(self):
        if self._edm is None:
            m = np.zeros((self.N, self.N, self.N))
            for node in range(self.N):
                for d in range(0, int(np.amax(self.dm) + 1)):
                    nodes = np.where(self.dm[node] == d)[0]
                    nodes_edges = [edge for n in nodes for edge in self.edges(n)]
                    for a, b in nodes_edges:
                        if m[node, a, b] == 0:
                            m[node, a, b] = d + 1
                            m[node, b, a] = d + 1
            m[m == 0] = np.inf
            self._edm = m
        return self._edm

    @property
    def common_neighbors_matrix(self):
        if self._common_neighbors_matrix is None:
            cnm = -1 * np.ones(self.M.shape)
            for node_a in range(self.N):
                neighbors_a = self.M[node_a]
                for node_b in np.where(neighbors_a > 0)[0]:
                    if cnm[node_a, node_b] != -1:
                        pass
                    else:
                        neighbors_b = self.M[node_b]
                        cnm[node_a, node_b] = np.sum(neighbors_a * neighbors_b)
                        cnm[node_b, node_a] = np.sum(neighbors_a * neighbors_b)
            cnm[cnm == -1] = 0
            self._common_neighbors_matrix = cnm
        return self._common_neighbors_matrix

    def __len__(self) -> int:
        if self._M is not None:
            return len(self._M)
        return len(self._G)  # type: ignore
