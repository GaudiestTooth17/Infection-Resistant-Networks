import sys
sys.path.append('')
from customtypes import Layout
from network import Network
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class SpatialConfiguration:
    """
    Contains a grid of agent locations and a diction of location on the grid to ID.

    The grid is 1 where an agent is present and 0 everywhere else. Its dtype is uint8.
    Agent IDs range from 0 to N-1 (inclusive) where N is the number of agents.
    """
    grid: np.ndarray
    agent_location_to_id: Dict[Tuple[int, int], int]

    @property
    def N(self) -> int:
        return len(self.agent_location_to_id)


class MakeLazySpatialNetwork:
    def __init__(self, config: SpatialConfiguration) -> None:
        """Can be called with different reaches to make the corresponding spatial network."""
        self._dm, self._layout = self._init_dist_matrix(config)
        self._N = config.N
        self._radius_to_network: Dict[int, Network] = {}

    def __call__(self, agent_reach: int) -> Network:
        if agent_reach not in self._radius_to_network:
            self._radius_to_network[agent_reach] = self._make_network_with_reach(agent_reach)
        return self._radius_to_network[agent_reach]

    @staticmethod
    def _init_dist_matrix(config: SpatialConfiguration)\
            -> Tuple[np.ndarray, Layout]:
        grid_shape = config.grid.shape
        distance_matrix = np.zeros(config.grid.shape)
        for (i, j), agent_id in config.agent_location_to_id.items():
            for (x, y), other_id in config.agent_location_to_id.items():
                distance_matrix[agent_id, other_id] = _distance(i, j, x, y)

        layout = {id_: (2*x/grid_shape[0]-1, 2*y/grid_shape[1]-1)
                  for (x, y), id_ in config.agent_location_to_id.items()}
        return distance_matrix, layout

    def _make_network_with_reach(self, agent_reach: int) -> Network:
        M = np.where(self._dm < agent_reach, 1, 0)
        return Network(M, layout=self._layout)


def make_random_spatial_configuration(grid_shape: Tuple[int, int], N: int, rng)\
        -> SpatialConfiguration:
    """
    Randomly populate an AgentGrid with N agents and the given shape using the
    provided RandomGenerator.
    """
    grid = np.zeros(grid_shape, dtype=np.uint8)
    loc_to_id = {}
    current_id = 0
    for _ in range(N):
        coord = tuple(rng.integers(N, size=2))
        while grid[coord] > 0:
            coord = tuple(rng.integers(N, size=2))
        grid[coord] = 1
        loc_to_id[coord] = current_id
        current_id += 1
    return SpatialConfiguration(grid, loc_to_id)


def _distance(x0: int, y0: int, x1: int, y1: int) -> float:
    return np.sqrt((x0-x1)**2 + (y0-y1)**2)
