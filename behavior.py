from typing import Sequence, Set, Tuple
import networkx as nx
import retworkx as rx
import numpy as np
from abc import ABC, abstractmethod, abstractproperty


class PressureHandler(ABC):
    """
    Handles which nodes are pressured.
    Shouldn't store any data beyond what was last called.
    """

    @abstractproperty
    def name(self) -> str:  # type: ignore
        pass

    @abstractmethod
    def __call__(self, sir: np.ndarray) -> np.ndarray:
        pass

    def __str__(self) -> str:
        return self.name


class UpdateConnections(ABC):

    def __init__(self, pressure_handler: PressureHandler) -> None:
        self._pressure_handler = pressure_handler
        self._last_pressured_nodes = None
        self._last_removed_edges = None
        self._last_diameter = None
        self._last_comps = None
        self._last_removed_edges = None
        self._last_perc_edges_removed: np.ndarray = None  # type: ignore

    @property
    def last_pressured_nodes(self) -> np.ndarray:
        """
        Return a True/False array where entries are True iff the node was
        pressured during the last step
        """
        return self._last_pressured_nodes  # type: ignore

    @property
    def last_removed_edges(self) -> np.ndarray:
        """
        As a matrix where 0 isn't touched but 1 is removed.
        """
        return self._last_removed_edges  # type: ignore

    @property
    def last_num_removed_edges(self) -> int:
        return np.sum(self.last_removed_edges > 0) // 2

    @property
    def last_diameter(self) -> int:
        return self._last_diameter  # type: ignore

    @property
    def last_comps(self) -> Tuple[Set[int]]:
        """
        List of components (which are a list of nodes)
        """
        return self._last_comps  # type: ignore

    @property
    def last_comp_sizes(self) -> Tuple[int, ...]:
        """
        List of ints as sizes of components
        """
        return tuple(map(lambda x: len(x), self.last_comps))

    @property
    def last_perc_edges_removed(self) -> np.ndarray:
        """
        Return an array where each entry is the percentage of that node's edges
        that were removed the last time the object was called
        """
        return self._last_perc_edges_removed

    @property
    def last_avg_comp_size(self) -> float:
        sizes = self.last_comp_sizes
        return sum(sizes) / len(sizes)

    @property
    def last_num_comps(self) -> int:
        return len(self.last_comps)

    def __call__(self, D: np.ndarray, M: np.ndarray, time_step: int, sir: np.ndarray) -> np.ndarray:
        pressured_nodes = self._pressure_handler(sir)
        D = self._call(D, M, time_step, pressured_nodes)
        R: np.ndarray = M - D  # type: ignore

        # Collect Data
        self._last_pressured_nodes = pressured_nodes
        self._last_removed_edges = R
        G = nx.from_numpy_array(D)
        self._last_comps = tuple(nx.connected_components(G))
        self._last_diameter = np.max(rx.distance_matrix(rx.networkx_converter(G)))
        self._last_perc_edges_removed = np.sum(R, axis=0) / np.sum(M, axis=0)

        return D

    def __str__(self) -> str:
        return self.name

    @abstractproperty
    def name(self) -> str:  # type: ignore
        pass

    @abstractmethod
    def _call(self, D: np.ndarray, M: np.ndarray, time_step: int,
              pressured_nodes: np.ndarray) -> np.ndarray:
        pass


"""
This is where the actual behaviors and pressure_handlers go.
"""


class NoMitigation(UpdateConnections):

    class NoPressure(PressureHandler):

        @property
        def name(self):
            return 'No Pressure'

        def __call__(self, sir: np.ndarray):
            return np.zeros(sir.shape[1], dtype=np.int32)

    def __init__(self):
        super().__init__(NoMitigation.NoPressure())

    @property
    def name(self) -> str:
        return 'No Mitigation'

    def _call(self, D: np.ndarray, M: np.ndarray, time_step: int,
              pressure_nodes: np.ndarray) -> np.ndarray:
        return M


class AllPressureHandler(PressureHandler):

    @property
    def name(self) -> str:
        return 'All Pressure Handler'

    def __call__(self, sir: np.ndarray) -> np.ndarray:
        """
        Returns every node as a true/false ndarray.
        """
        return np.ones(sir.shape[1], dtype=np.int32)


class DistancePressureHandler(PressureHandler):
    def __init__(self, DM: np.ndarray, distance: int):
        """
        Pressure is determined based on the given distance and distance matrix.
        """
        self.DM = DM
        self.distance = distance

    @property
    def name(self) -> str:
        return 'Distance Pressure Handler'

    def __call__(self, sir: np.ndarray) -> np.ndarray:
        """
        Returns every node within the specified distance of any
        infectious node as a true/false ndarray.
        """
        infectious_agents = sir[1] > 0
        return np.sum(self.DM[infectious_agents] <= self.distance, axis=0) > 0


class MultiPressureHandler(PressureHandler):
    def __init__(self, pressure_handlers: Tuple[PressureHandler]):
        self.pressure_handlers = pressure_handlers
        """
        Takes multiple pressure handlers and returns all of the pressured nodes among any of them
        as a true/false ndarray.
        """

    @property
    def name(self) -> str:
        n = 'OrPressureHandler: '
        for p in self.pressure_handlers:
            n += p.name + ' '
        return n

    def __call__(self, sir: np.ndarray) -> np.ndarray:

        pressured_nodes = np.zeros(sir.shape[1], dtype=np.int32)
        for p in self.pressure_handlers:
            pressured_nodes += p(sir)
        return pressured_nodes > 0


class BetweenDistancePressureHandler(PressureHandler):
    def __init__(self, DM: np.ndarray, min_distance: int, max_distance):
        """
        Pressure is determined based on the given distances and distance matrix.
        """
        self.DM = DM
        self.min_distance = min_distance
        self.max_distance = max_distance

    @property
    def name(self) -> str:
        return 'Distance Pressure Handler'

    def __call__(self, sir: np.ndarray) -> np.ndarray:
        """
        Returns every node within the specified distance of any
        infectious node as a true/false ndarray.
        """
        infectious_agents = sir[1] > 0
        greater_than_min = self.min_distance <= self.DM[infectious_agents]
        less_than_max = self.DM[infectious_agents] < self.max_distance
        return np.sum((greater_than_min * less_than_max), axis=0) > 0


class FlickerPressureBehavior(UpdateConnections):
    def __init__(self, rng,
                 pressure_handler: PressureHandler,
                 flicker_probability: float = .25):
        """
        Agents receive pressure when nearby agents become infectious. Agents
        with enough pressure will flicker connections to nearby agents.
        """
        super().__init__(pressure_handler)
        self._rng = rng
        self._flicker_probability = flicker_probability

    @property
    def name(self) -> str:
        return f'Flicker Pressure Behavior ({self._pressure_handler.name})'

    def _call(self, D: np.ndarray, M: np.ndarray, time_step: int,
              pressured_nodes: np.ndarray) -> np.ndarray:
        flicker_agents = (pressured_nodes & (self._rng.random(len(D)) < self._flicker_probability))
        R = np.copy(M)
        R[flicker_agents, :] = 0
        R[:, flicker_agents] = 0

        return R


class MultiPressureBehavior(UpdateConnections):
    def __init__(self, rng,
                 behaviors: Sequence[UpdateConnections]):
        # It's okay to pass None in here because we redefined __call__
        super().__init__(None)  # type: ignore
        self._rng = rng
        self._behaviors = behaviors

    @property
    def name(self) -> str:
        return 'MultiPressureBehavior'

    def __call__(self, D: np.ndarray, M: np.ndarray, time_step: int, sir: np.ndarray) -> np.ndarray:
        """
        Since __call__ was redefined, we don't need _call.
        """
        final_ND = np.zeros(D.shape, dtype=bool)
        final_pressured_nodes = np.zeros(D.shape[0], dtype=bool)
        for behavior in self._behaviors:
            pressured_nodes = behavior._pressure_handler(sir)
            ND = behavior._call(D, M, time_step, pressured_nodes)
            final_ND = final_ND | (ND > 0)
            final_pressured_nodes = final_pressured_nodes | pressured_nodes

        # Collect Data
        R: np.ndarray = M - final_ND  # type: ignore
        self._last_pressured_nodes = final_pressured_nodes
        self._last_removed_edges = R
        G = nx.from_numpy_array(final_ND)
        self._last_comps = tuple(nx.connected_components(G))
        self._last_diameter = np.max(rx.distance_matrix(rx.networkx_converter(G)))
        self._last_perc_edges_removed = np.sum(R, axis=0) / np.sum(M, axis=0)

        return ND  # type: ignore

    def _call(self, D: np.ndarray, M: np.ndarray, time_step: int, pressured_nodes: np.ndarray)\
            -> np.ndarray:
        return super()._call(D, M, time_step, pressured_nodes)
