import sys
sys.path.append('')
from typing import Union, Tuple, Dict, Optional
from dataclasses import dataclass
from customtypes import Number, Layout, NodeColors
import time
from analyzer import visualize_network
from scipy.sparse import dok_matrix
import numpy as np
import networkx as nx
from partitioning import intercommunity_edges_to_communities, fluidc_partition
from tqdm import tqdm
import itertools as it
RAND = np.random.default_rng()


@dataclass(unsafe_hash=True, frozen=True)
class Agent:
    color: Union[str, Tuple[int, int, int]]
    reach: Number


def social_circles_entry_point():
    # if len(sys.argv) < 2:
    #     print(f'Usage: {sys.argv[0]} <output name>')
    #     return
    rand = np.random.default_rng(0)
    num_agents = 250
    num_purple = int(num_agents * .1)
    num_blue = int(num_agents * .2)
    num_green = num_agents - num_purple - num_blue
    grid_dim = int(num_agents+100)  # the denominator is the desired density
    print(f'Density = {num_agents/(grid_dim**2)}')

    agents = {Agent('green', 30): num_green,
              Agent('blue', 40): num_blue,
              Agent('purple', 50): num_purple}
    for _ in range(100):
        start_time = time.time()
        social_circles_result = make_social_circles_network(agents, (grid_dim, grid_dim),
                                                            verbose=False, rand=rand)
        if social_circles_result is None:
            print('Generation failed.')
            exit(1)
        G, layout, _ = social_circles_result
        print(f'Finished social circles network ({time.time() - start_time}s).')
        visualize_network(G, layout, 'Social Circles', block=False)
        num_communities = len(G) // 20
        print(f'Aiming for {num_communities} communities.')
        to_remove = fluidc_partition(G, num_communities)
        # somehow this frequently splits G into more components than I request
        G.remove_edges_from(to_remove)
        print(f'G has {nx.number_connected_components(G)} connected components.')
        visualize_network(G, layout, 'Partitioned')
        print('\n\n')


def make_social_circles_network(agent_type_to_quantity: Dict[Agent, int],
                                grid_size: Tuple[int, int],
                                none_on_disconnected: bool = False,
                                verbose: bool = False,
                                max_tries: int = 5,
                                rand=RAND)\
        -> Optional[Tuple[nx.Graph, Layout, NodeColors]]:
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
                x, y = choose_empty_spot(grid, rand)
                grid[x, y] = agent.reach
                new_agents.append((x, y))
                loc_to_id[(x, y)] = current_id
                current_id += 1
            if verbose:
                new_agents = tqdm(new_agents)
                print('Connecting agents.')
            for x, y in new_agents:
                neighbors = search_for_neighbors(grid, x, y)
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
            return G, layout, colors
        elif verbose:
            print(f'Finished {attempt+1} tries.')

    # return None to signal failure
    return None


def choose_empty_spot(grid, rand) -> Tuple[int, int]:
    x, y = rand.integers(grid.shape[0]), rand.integers(grid.shape[1])
    while grid[x, y] > 0:
        x, y = rand.integers(grid.shape[0]), rand.integers(grid.shape[1])
    return x, y


def search_for_neighbors(grid, x, y):
    reach = grid[x, y]
    min_x = max(0, x-reach)
    max_x = min(grid.shape[0]-1, x+reach)
    min_y = max(0, y-reach)
    max_y = min(grid.shape[1]-1, y+reach)
    neighbors = {(i, j)
                 for (i, j) in it.product(range(min_x, max_x),
                                          range(min_y, max_y))
                 if all((grid[i, j] > 0,
                         distance(x, y, i, j) <= reach,
                         (x, y) != (i, j)))}
    return neighbors


def distance(x0, y0, x1, y1) -> float:
    return np.sqrt((x0-x1)**2 + (y0-y1)**2)


if __name__ == '__main__':
    social_circles_entry_point()
