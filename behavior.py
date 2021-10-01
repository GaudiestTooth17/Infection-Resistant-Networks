from dataclasses import dataclass
from typing import Callable, Collection, List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from customtypes import Layout
from network import Network
from socialgood import get_distance_matrix
from abc import ABC, abstractmethod, abstractproperty


UpdateConnections = Callable[[np.ndarray, np.ndarray, int, np.ndarray], np.ndarray]
"""
Update the dynamic matrix D.
Parameters are D, M, current step, current sir array
"""

PressureHandler = Callable[[np.ndarray], np.ndarray]
"""
Gets the pressured nodes.
Parameters are current sir array.
"""


class PressureHandler(ABC):

    @abstractproperty
    def name(self) -> str:
        pass

    @abstractmethod
    def last_pressured_nodes(self) -> np.ndarray:
        pass

    @abstractmethod
    def __call__(self, sir: np.ndarray) -> np.ndarray:
        pass

    def __str__(self) -> str:
        return self.name


class UpdateConnections(ABC):

    @abstractproperty
    def name(self) -> str:
        pass

    @abstractproperty
    def last_pressured_nodes(self) -> np.ndarray:
        pass

    @abstractproperty
    def last_removed_edges(self) -> np.ndarray:
        """
        As a matrix where 0 isn't touched but 1 is removed.
        """
        pass

    def last_num_removed_edges(self) -> int:
        return len(self.last_removed_edges)

    @abstractproperty
    def last_diameter(self) -> int:
        pass

    @abstractproperty
    def last_comps(self) -> List[List[int]]:
        """
        List of components (which are a list of nodes)
        """
        pass

    def last_comp_sizes(self) -> List[int]:
        """
        List of ints as sizes of components
        """
        return map(lambda x: len(x))

    def last_avg_comp_size(self) -> int:
        sizes = self.last_comp_sizes
        return sum(sizes) / len(sizes)

    def last_num_comps(self) -> int:
        return len(self.last_comps)

    @abstractmethod
    def __call__(self, sir: np.ndarray) -> np.ndarray:
        pass

    def __str__(self) -> str:
        return self.name


class NoMitigation:
    def __call__(self, D, M, time_step, sir):
        return M


class AllPressureHandler:
    def __init__(self) -> None:
        """
        Handles which nodes are pressured, etc.
        Pressures all nodes.
        """
        self.distance = np.inf
        self.num_pressured_nodes = []

    def __call__(self, sir: np.ndarray) -> np.ndarray:
        """
        Returns every node as a true/false ndarray.
        """
        self.num_pressured_nodes.append(sir.shape[1])
        return np.zeros(sir.shape[1]) == 0


class DistancePressureHandler:
    def __init__(self, DM: np.ndarray, distance: int):
        """
        Handles which nodes are pressured, etc.
        Pressure is determined based on edge distance.
        """
        self.DM = DM
        self.distance = distance

    def __call__(self, sir: np.ndarray) -> np.ndarray:
        """
        Returns every node within the specified distance of any
        infectious node as a true/false ndarray.
        """
        infectious_agents = sir[1] > 0
        pressured_nodes = np.sum(self.DM[infectious_agents] <= self.distance, axis=0) > 0
        return pressured_nodes


class FlickerPressureBehavior:
    def __init__(self, net: Network,
                 rng,
                 pressure_handler: DistancePressureHandler,
                 flicker_probability: float = .25):
        """
        Agents receive pressure when nearby agents become infectious. Agents
        with enough pressure will remove connections to nearby agents.
        """
        self._name = f'SimplePressure(radius={pressure_handler.distance}, flicker_probability={flicker_probability})'
        self._pressure_handler = pressure_handler
        self._rng = rng
        self._flicker_probability = flicker_probability
        self._last_pressured_nodes = None
        self._last_removed_edges = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def pressured_nodes(self) -> int:
        return self._last_pressured_nodes

    @property
    def removed_edges(self) -> int:
        return self._last_removed_edges

    def __call__(self, D: np.ndarray, M: np.ndarray, time_step: int, sir: np.ndarray) -> np.ndarray:
        pressured_nodes = self._pressure_handler(sir)
        self._last_num_pressured_nodes = np.sum(pressured_nodes)

        flicker_agents = (pressured_nodes & (self._rng.random(len(D)) < self._flicker_probability))
        R = np.copy(M)
        R[flicker_agents, :] = 0
        R[:, flicker_agents] = 0

        self._last_num_removed_edges = int(np.sum(M - R) / 2)

        return R


class FlickerBehavior:
    def __init__(self, M: np.ndarray,
                 edges_to_flicker: Collection[Tuple[int, int]],
                 flicker_pattern: Sequence[bool],
                 name: Optional[str] = None) -> None:
        """
        Flickers inter-community edges according to flicker_pattern.

        M: The original network
        edges_to_flicker: The edges of the network that will be toggled.
        flicker_pattern: True means that inter-community edges are on. False means they are off.
                         The values will automatically cycle after they have all been used.
        """
        self._flicker_pattern = flicker_pattern
        self._edges_on_M = np.copy(M)
        self._edges_off_M = np.copy(M)
        for u, v in edges_to_flicker:
            self._edges_off_M[u, v] = 0
            self._edges_off_M[v, u] = 0

        self.name = name if name is not None else f'Flicker {flicker_pattern}'

    def __call__(self, D: np.ndarray, M: np.ndarray, time_step: int, sir: np.ndarray) -> np.ndarray:
        if self._flicker_pattern[time_step % len(self._flicker_pattern)]:
            return self._edges_on_M
        return self._edges_off_M


class RandomFlickerBehavior:
    def __init__(self, M: np.ndarray,
                 edges_to_flicker: Collection[Tuple[int, int]],
                 flicker_probability: float,
                 rng,
                 name: Optional[str] = None) -> None:
        """
        Flickers inter-community edges according to flicker_pattern.

        M: The original network
        edges_to_flicker: The edges of the network that will be toggled.
        flicker_probability: The probability that ALL of the edges will be present at a step.
                             To be extra clear, the edges are either all present or all absent.
        """
        self._flicker_probability = flicker_probability

        self._edges_on_M = np.copy(M)
        self._edges_off_M = np.copy(M)
        for u, v in edges_to_flicker:
            self._edges_off_M[u, v] = 0
            self._edges_off_M[v, u] = 0

        self._rng = rng
        self.name = name if name is not None else f'Flicker {flicker_probability}'

    def __call__(self, D: np.ndarray, M: np.ndarray, time_step: int, sir: np.ndarray) -> np.ndarray:
        if self._rng.random() < self._flicker_probability:
            return self._edges_on_M
        return self._edges_off_M


class SimpleEdgePressureBehavior:
    def __init__(self, net: Network,
                 rng,
                 radius: int = 3,
                 flicker_probability: float = .25):
        """
        Edges recieve pressure when nearby agents become infectious. Edges with enough pressure
        will "dissapear".
        """
        self._radius = radius
        self._name = f'SimplePressure(radius={radius}, flicker_probability={flicker_probability})'
        self._pressure = {}
        self._pressure.update({(a, b): 0 for a, b in net.edges})
        self._pressure.update({(b, a): 0 for a, b in net.edges})
        self._flicker_probability = flicker_probability
        self._rng = rng
        self._edm = net.edm

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, D: np.ndarray, M: np.ndarray, time_step: int, sir: np.ndarray) -> np.ndarray:
        infectious_agents = sir[1] == 1
        if infectious_agents.any():
            pressured_edges = zip(*np.where(self._edm[infectious_agents] <= self._radius)[1:])
            for edge in pressured_edges:
                # print(edge)
                self._pressure[edge] += 1

        recovered_agents = sir[2] == 1
        if recovered_agents.any():
            unpressured_edges = zip(*np.where(self._edm[recovered_agents] <= self._radius)[1:])
            for edge in unpressured_edges:
                self._pressure[edge] -= 1
        R = np.copy(M)
        for (a, b), p in self._pressure.items():
            if p > 0 and self._rng.random() < self._flicker_probability:
                R[a, b] = 0
                R[b, a] = 0
        return R


class UnifiedPressureFlickerBehavior:
    def __init__(self, net: Network,
                 rng,
                 radius: int = 3,
                 name: Optional[str] = None):
        """
        Agents receive pressure when nearby agents become infectious. Agents
        with enough pressure will remove connections to nearby agents.
        """
        self._net = net
        self._radius = radius
        self._name = f'Pressure(radius={radius})' if name is None else name
        self._dm = get_distance_matrix(net)
        self._pressure = np.zeros(net.N)
        self._flicker_probability = 0.25
        self._rng = rng

    def __call__(self, D: np.ndarray, M: np.ndarray, time_step: int, sir: np.ndarray) -> np.ndarray:
        infectious_agents = sir[1] == 1
        if infectious_agents.any():
            pressured_agents = (self._dm[infectious_agents] <= self._radius)[0]
            self._pressure[pressured_agents] += 1

        recovered_agents = sir[2] == 1
        if recovered_agents.any():
            unpressured_agents = (self._dm[recovered_agents] <= self._radius)[0]
            self._pressure[unpressured_agents] -= 1

        R = np.copy(M)
        if self._rng.random() > self._flicker_probability:
            flicker_agents = self._pressure > 0

        # flicker_agents = ((self._pressure > 0) & (self._rng.random(self._pressure.shape)
        #                                           < self._flicker_probability))
            R[flicker_agents, :] = 0
            R[:, flicker_agents] = 0
        # print('Edges Removed', (np.sum(M) - np.sum(R)) / 2)
        return R


class PressureDecayBehavior:
    def __init__(self, net: Network,
                 rng,
                 radius: int = 3,
                 name: Optional[str] = None):
        """
        Agents receive pressure when nearby agents become infectious. Agents
        with enough pressure will remove connections to nearby agents.
        """
        self._net = net
        self._radius = radius
        self._name = f'Pressure(radius={radius})' if name is None else name
        self._dm = get_distance_matrix(net)
        self._pressure = np.zeros(net.N)
        self._flicker_probability = rng.random(net.N)
        self._rng = rng

    def __call__(self, D: np.ndarray, M: np.ndarray, time_step: int, sir: np.ndarray) -> np.ndarray:
        infectious_agents = sir[1] > 0
        if infectious_agents.any():
            pressured_agents = (self._dm[infectious_agents] <= self._radius)[0]
            self._pressure[pressured_agents] += self._flicker_probability[pressured_agents]

        flicker_agents = ((self._pressure >= .5) & (self._rng.random(self._pressure.shape)
                                                    < self._flicker_probability))
        R = np.copy(M)
        R[flicker_agents, :] = 0
        R[:, flicker_agents] = 0
        # print('Edges Removed', (np.sum(M) - np.sum(R)) / 2)

        self._pressure = self._pressure * self._flicker_probability

        return R


class PressureFlickerBehavior:
    def __init__(self, net: Network,
                 rng,
                 radius: int = 3,
                 name: Optional[str] = None):
        """
        Agents receive pressure when nearby agents become infectious. Agents
        with enough pressure will remove connections to nearby agents.
        """
        self._net = net
        self._radius = radius
        self._name = f'Pressure(radius={radius})' if name is None else name
        self._dm = get_distance_matrix(net)
        self._pressure = np.zeros(net.N)
        # self._flicker_probability = rng.random(net.N)
        self._flicker_probability = rng.power(9, net.N)
        self._flicker_threshold = (1 - rng.power(4, net.N))
        self._flicker_pressure_multiplier = self._flicker_probability
        self._pressure_increase_rate = self._flicker_probability
        self._pressure_decay = self._flicker_probability
        self._rng = rng

    def __call__(self, D: np.ndarray, M: np.ndarray, time_step: int, sir: np.ndarray) -> np.ndarray:
        infectious_agents = sir[1] > 0
        if infectious_agents.any():
            pressured_agents = (self._dm[infectious_agents] <= self._radius)[0]
            self._pressure[pressured_agents] += self._pressure_increase_rate[pressured_agents]

        flicker_amount = self._pressure / self._pressure_to_flicker
        current_flicker_prob = 1 - np.minimum((1 - self._flicker_probability),
                                              self._flicker_probability) ** flicker_amount

        flicker_agents = ((self._pressure > self._pressure_to_flicker) &
                          (self._rng.random(self._pressure.shape) < current_flicker_prob))
        R = np.copy(M)
        R[flicker_agents, :] = 0
        R[:, flicker_agents] = 0

        self._pressure = self._pressure * self._pressure_decay

        return R
