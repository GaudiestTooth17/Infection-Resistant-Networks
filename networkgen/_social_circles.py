import sys
sys.path.append('')
from network import Network
from typing import Union, Tuple, Dict, Optional
from dataclasses import dataclass
from customtypes import Number, NodeColors
from scipy.sparse import dok_matrix
import numpy as np
import networkx as nx
from tqdm import tqdm
import itertools as it
from numba import njit
import fileio as fio


@dataclass(unsafe_hash=True, frozen=True)
class Agent:
    color: Union[str, Tuple[int, int, int]]
    reach: Number


def social_circles_entry_point():
    rng = np.random.default_rng(0)
    N = 1000
    n_green = N // 3
    n_blue = N // 3
    n_purple = N - n_green - n_blue
    grid_dim = 225
    print(f'Density = {N/(grid_dim**2)}')

    agents = {Agent('green', 30): n_green,
              Agent('blue', 40): n_blue,
              Agent('purple', 50): n_purple}
    nets = [make_social_circles_network(agents, (grid_dim, grid_dim), verbose=False,
                                        rng=rng, max_tries=1)[0]  # type: ignore
            for _ in tqdm(range(100))]
    fio.write_network_class('SocialCircles', nets)


def make_social_circles_network(agent_type_to_quantity: Dict[Agent, int],
                                grid_size: Tuple[int, int],
                                none_on_disconnected: bool = False,
                                *,
                                verbose: bool = False,
                                max_tries: int = 5,
                                rng)\
        -> Optional[Tuple[Network, NodeColors]]:
    """Return a social circles network or None on timeout."""
    for attempt in range(max_tries):
        agents = sorted(agent_type_to_quantity.items(),
                        key=lambda agent_quantity: agent_quantity[0].reach,
                        reverse=True)
        try:
            grid = np.zeros(grid_size, dtype='uint8')
        except MemoryError:
            print('Warning: Not enough memory. Switching to dok_matrix.', file=sys.stderr)
            grid = dok_matrix(grid_size, dtype='uint8')
        num_nodes = sum(agent_type_to_quantity.values())
        M = np.zeros((num_nodes, num_nodes), dtype='uint8')
        # place the agents with the largest reach first
        loc_to_id = {}
        current_id = 0
        for agent, quantity in agents:
            new_agents = []
            if verbose:
                print(f'Placing agents with reach {agent.reach}.')
                range_quantity = tqdm(range(quantity))
            else:
                range_quantity = range(quantity)
            for _ in range_quantity:
                x, y = choose_empty_spot(grid, rng)
                grid[x, y] = agent.reach
                new_agents.append((x, y))
                loc_to_id[(x, y)] = current_id
                current_id += 1
            if verbose:
                new_agents = tqdm(new_agents)
                print('Connecting agents.')
            for x, y in new_agents:
                neighbors = fast_search_for_neighbors(grid, x, y)
                for i, j in neighbors:
                    M[loc_to_id[(x, y)], loc_to_id[(i, j)]] = 1
                    M[loc_to_id[(i, j)], loc_to_id[(x, y)]] = 1

        colors = []
        for agent, quantity in agent_type_to_quantity.items():
            colors += [agent.color]*quantity
        layout = {id_: (2*x/grid_size[0]-1, 2*y/grid_size[1]-1)
                  for (x, y), id_ in loc_to_id.items()}
        G = nx.Graph(M)
        # return the generated network if it is connected or if we don't care
        if (not none_on_disconnected) or nx.is_connected(G):
            if verbose:
                print(f'Success after {attempt+1} tries.')
            return Network(G, layout=layout), colors
        elif verbose:
            print(f'Finished {attempt+1} tries.')

    # return None to signal failure
    return None


def choose_empty_spot(grid, rand) -> Tuple[int, int]:
    x, y = rand.integers(grid.shape[0]), rand.integers(grid.shape[1])
    while grid[x, y] > 0:
        x, y = rand.integers(grid.shape[0]), rand.integers(grid.shape[1])
    return x, y


def search_for_neighbors(grid, x: int, y: int):
    reach = grid[x, y]
    # The connections on the grid need to wrap around,
    # so we mod the values by the grid  dims
    min_x = (x-reach) % grid.shape[0]
    max_x = (x+reach) % grid.shape[0]
    min_y = (y-reach) % grid.shape[1]
    max_y = (y+reach) % grid.shape[1]
    # Because the values got modded, the mins might actually be higher than the maxes,
    # so we reassign the values here to make sure that the values are properly labeled
    min_x, max_x = sorted((min_x, max_x))
    min_y, max_y = sorted((min_y, max_y))

    neighbors = {(i, j)
                 for (i, j) in it.product(range(min_x, max_x),
                                          range(min_y, max_y))
                 if all((grid[i, j] > 0,
                         distance(x, y, i, j) <= reach,
                         (x, y) != (i, j)))}
    return neighbors


@njit
def fast_search_for_neighbors(grid, x: int, y: int):
    reach = grid[x, y]
    # The connections on the grid need to wrap around,
    # so we mod the values by the grid  dims
    min_x = (x-reach) % grid.shape[0]
    max_x = (x+reach) % grid.shape[0]
    min_y = (y-reach) % grid.shape[1]
    max_y = (y+reach) % grid.shape[1]
    # Because the values got modded, the mins might actually be higher than the maxes,
    # so we reassign the values here to make sure that the values are properly labeled
    min_x, max_x = sorted((min_x, max_x))
    min_y, max_y = sorted((min_y, max_y))

    neighbors = []
    for i in range(min_x, max_x+1):
        for j in range(min_y, max_y+1):
            if grid[i, j] > 0 and distance(x, y, i, j) <= reach and (x, y) != (i, j):
                neighbors.append((i, j))
    return neighbors


@njit
def distance(x0: int, y0: int, x1: int, y1: int) -> float:
    return np.sqrt((x0-x1)**2 + (y0-y1)**2)


if __name__ == '__main__':
    social_circles_entry_point()
