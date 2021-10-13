import sys
sys.path.append('')
from dataclasses import dataclass
from network import Network
from sim_dynamic import (Disease, make_starting_sir, simulate)
from behavior import UpdateConnections
from typing import (Any, Callable, Collection, List, Optional, Tuple, TypeVar,
                    Sequence, Dict, Union)
from abc import ABC, abstractmethod
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.stats import wasserstein_distance
import os
import csv
from customtypes import Array, Number
from networkgen import make_connected_community_network
from pathlib import Path
import copy
import fileio as fio
from scipy.stats import entropy
import itertools as it
T = TypeVar('T')


def calc_entropy(a: np.ndarray, bins: int) -> float:
    hist, _ = np.histogram(a, bins=bins)
    return entropy(hist)


class MakeNetwork(ABC):
    """
    Interface for classes that create networks and keep track of the type of network they create.
    """
    @property
    @abstractmethod
    def class_name(self) -> str:
        """A description of the class of random network, or just a name for a non random network."""
        pass

    @property
    @abstractmethod
    def seed(self) -> Optional[int]:
        """
        The seed used to make the random network. This will be the seed for
        default_rng if it is a custom random network, or the seed passed to the
        NetworkX function. None is returned for networks that aren't generated
        randomly or if the seed was not set.
        """
        return None

    @abstractmethod
    def __call__(self) -> Network:
        """Return the appropriate network."""
        pass


class MakeConnectedCommunity(MakeNetwork):
    def __init__(self, community_size: int, inner_bounds: Tuple[int, int],
                 num_comms: int, outer_bounds: Tuple[int, int], seed: Optional[int] = None):
        self._community_size = community_size
        self._inner_bounds = inner_bounds
        self._num_comms = num_comms
        self._outer_bounds = outer_bounds
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._class_name = f'ConnComm(N_comm={community_size},ib={inner_bounds},'\
                           f'num_comms={num_comms},ob={outer_bounds})'

    @property
    def class_name(self) -> str:
        return self._class_name

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    def __call__(self) -> Network:
        id_dist = self._rng.integers(self._inner_bounds[0], self._inner_bounds[1],
                                     self._community_size, endpoint=True)
        if np.sum(id_dist) % 2 > 0:
            id_dist[np.argmin(id_dist)] += 1
        od_dist = self._rng.integers(self._outer_bounds[0], self._outer_bounds[1],
                                     self._num_comms, endpoint=True)
        if np.sum(od_dist) % 2 > 0:
            od_dist[np.argmin(od_dist)] += 1

        net = make_connected_community_network(id_dist, od_dist, self._rng)
        if net is None:
            raise Exception('This should not have happened.')
        return net


class MakeBarabasiAlbert(MakeNetwork):
    def __init__(self, N: int, m: int, seed: Optional[int] = None):
        self._N = N
        self._m = m
        self._seed = seed
        self._class_name = f'BarabasiAlbert(N={N},m={m})'

    @property
    def class_name(self) -> str:
        return self._class_name

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    def __call__(self) -> Network:
        return Network(nx.barabasi_albert_graph(self._N, self._m, self._seed))


class MakeWattsStrogatz(MakeNetwork):
    def __init__(self, N: int, k: int, p: float, seed: Optional[int] = None):
        self._N = N
        self._k = k
        self._p = p
        self._seed = seed
        self._class_name = f'WattsStrogatz(N={N},k={k},p={p})'

    @property
    def class_name(self) -> str:
        return self._class_name

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    def __call__(self) -> Network:
        return Network(nx.watts_strogatz_graph(self._N, self._k, self._p, self._seed))


class MakeErdosRenyi(MakeNetwork):
    def __init__(self, N: int, p: float, seed: Optional[int] = None):
        self._N = N
        self._p = p
        self._seed = seed
        self._class_name = f'ErdosRenyi(N={N},p={p})'

    @property
    def class_name(self) -> str:
        return self._class_name

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    def __call__(self) -> Network:
        return Network(nx.erdos_renyi_graph(self._N, self._p, self._seed))


class MakeGrid(MakeNetwork):
    def __init__(self, m: int, n: int) -> None:
        self._m = m
        self._n = n
        self._class_name = f'Grid(n={n},m={m})'
        # There is no randomness, so just save the network once it is generated once.
        self._net: Optional[Network] = None

    @property
    def class_name(self) -> str:
        return self._class_name

    @property
    def seed(self) -> None:
        return None

    def __call__(self) -> Network:
        if self._net is None:
            self._net = Network(nx.grid_2d_graph(self._m, self._n))
        # Make a copy so that if someone really wants to mutate the Network, it
        # won't screw up future return values.
        return copy.deepcopy(self._net)


class LoadNetwork(MakeNetwork):
    def __init__(self, name: str):
        self._name = name
        self._net = None

    @property
    def class_name(self) -> str:
        return self._name

    @property
    def seed(self) -> None:
        return None

    def __call__(self) -> Network:
        if self._net is None:
            path = fio.network_names_to_paths((self._name,))[0]
            self._net = fio.read_network(path)
        return self._net
