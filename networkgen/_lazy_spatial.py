import sys
sys.path.append('')
from network import Network
import numpy as np
from typing import Dict, Set, Tuple
import itertools as it


class LazySpatialNetwork:
    def __init__(self, grid_shape: Tuple[int, int], N: int, rng) -> None:
        self._agent_grid = self._generate_agent_layout(grid_shape, N, rng)
        self._N = N
        self._radius_to_network: Dict[int, Network] = {}

    def __get_item__(self, item: int) -> Network:
        if item not in self._radius_to_network:
            pass
        return self._radius_to_network

    def _generate_agent_layout(self, grid_shape: Tuple[int, int], N: int, rng) -> np.ndarray:
        grid = np.zeros(grid_shape, dtype=np.uint8)
        for _ in range(N):
            coord = tuple(rng.integers(N, size=2))
            while grid[coord] > 0:
                coord = tuple(rng.integers(N, size=2))
            grid[coord] = 1
        return grid

    def _make_network_with_reach(self, agent_reach: int) -> Network:
        M = np.zeros((self._N, self._N), dtype=np.uint8)
        loc_to_id = {}
        current_id = 0
        for agent in range(self._N):
            pass

    def _search_for_neighbors(self, x: int, y: int) -> Set[Tuple[int, int]]:
        reach = self._agent_grid[x, y]
        min_x = max(0, x-reach)
        max_x = min(self._agent_grid.shape[0]-1, x+reach)
        min_y = max(0, y-reach)
        max_y = min(self._agent_grid.shape[1]-1, y+reach)
        neighbors = {(i, j)
                     for (i, j) in it.product(range(min_x, max_x),
                                              range(min_y, max_y))
                     if all((self._agent_grid[i, j] > 0,
                            _distance(x, y, i, j) <= reach,
                            (x, y) != (i, j)))}
        return neighbors


def _distance(x0: int, y0: int, x1: int, y1: int) -> float:
    return np.sqrt((x0-x1)**2 + (y0-y1)**2)
