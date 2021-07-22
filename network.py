from typing import Union, Optional, Collection, Tuple, Iterable
from customtypes import Communities
import networkx as nx
import numpy as np
from partitioning import fluidc_partition, intercommunity_edges_to_communities


class Network:
    def __init__(self, data: Union[nx.Graph, np.ndarray],
                 intercommunity_edges: Optional[Collection[Tuple[int, int]]] = None,
                 communities: Optional[Communities] = None,
                 community_size: int = 25) -> None:
        """
        Holds both the NetworkX and NumPy representation of a network. It starts
        off with just one and lazily creates the other representation.

        Caveats:
        Do not mutate; changes in one data structure are not reflected in the other.
        Does not support selfloops or multiedges.
        """
        if isinstance(data, nx.Graph):
            self._G = data
            self._M = None
        else:
            self._G = None
            self._M = data
        self._intercommunity_edges = intercommunity_edges
        self._communities = communities
        self._community_size = community_size

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
    def N(self) -> int:
        return self.__len__()

    @property
    def E(self) -> int:
        if self._G is not None:
            return len(self._G.edges)
        return np.sum(self._M > 0) // 2  # type: ignore

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

    def __len__(self) -> int:
        if self._M is not None:
            return len(self._M)
        return len(self._G)  # type: ignore
