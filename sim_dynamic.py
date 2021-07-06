import time
from typing import Callable, Collection, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from fileio import get_network_name, read_network
from customtypes import Layout, ExperimentResults
from multiprocessing import Pool
import os
from numba import njit
RAND = np.random.default_rng()

UpdateConnections = Callable[[np.ndarray, np.ndarray, int, np.ndarray], np.ndarray]
"""
Update the dynamic matrix D.
Parameters are D, M, current step, current sir array
"""


@dataclass
class Disease:
    days_infectious: int
    trans_prob: float


def main():
    start_time = time.time()
    # network_names = ('agent-generated-500',
    #                  'annealed-agent-generated-500',
    #                  'annealed-large-diameter',
    #                  'annealed-medium-diameter',
    #                  'annealed-short-diameter',
    #                  'cgg-500',
    #                  'watts-strogatz-500-4-.1',
    #                  'elitist-500',
    #                  'spatial-network')
    network_names = ('connected-comm-50-10',)
    network_paths = ['networks/'+name+'.txt' for name in network_names]
    # verify that all the networks exist
    found_errors = False
    for path in network_paths:
        if not os.path.isfile(path):
            print(f'{path} does not exist!')
            found_errors = True
    if found_errors:
        print('Fix errors before continuing')
        exit(1)

    flicker_configurations = [FlickerConfig((True,), 'Static'),
                              FlickerConfig((True, False), 'One Half Flicker'),
                              FlickerConfig((True, False, False), 'One Third Flicker')]
    arguments = [(path, 1000, 500, Disease(4, .2), flicker_configurations)
                 for path in network_paths]
    # use a maximum of 10 cores
    with Pool(min(len(arguments), 10)) as p:
        expirement_results = p.map(run_experiments, arguments)

    results_dir = 'experiment results'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    for result in expirement_results:
        if result is not None:
            result.save(results_dir)

    print(f'Finished simulations ({time.time()-start_time}).')


def pool_friendly_simulate(args):
    M, n_to_infect, disease, behavior, max_steps = args
    sir0 = make_starting_sir(M.shape[0], n_to_infect)
    return simulate(M, sir0, disease, behavior, max_steps, None)[-1]


def simulate(M: np.ndarray,
             sir0: np.ndarray,
             disease: Disease,
             update_connections: UpdateConnections,
             max_steps: int,
             layout: Optional[Layout]) -> List[np.ndarray]:
    """
    Simulate an infection on a dynamic network.

    The simulation will end early if there are no susceptible agents left or if there are no
    infectious agents left. In the first case, the returned list will be the same length as
    the number of steps taken. In the second case, the list will have length == max_steps and the
    entries after the current step will be the same as the entry at the current step. This is so
    that an objective function can use the length of the returned list.

    M: The base network.
    sir0: The initial states of the agents. It has shape (3, N). The first dimension is for each
          state in SIR. The second dimension is for an agent. A 0 entry means that the agent is not
          in that state. A positive entry means that the agent has spent 1 fewer days than the
          number in that state.
    disease: The Disease to simulate.
    update_connections: A function that updates the dynamic adjacency matrix.
    max_steps: The maximum number of steps to run the simulation for before returning.
    layout: If you want visualization, provide a layout to use. Pass None for no visualization.
    """
    rand = RAND  # increase access speed by making a local reference
    sirs: List[np.ndarray] = [None] * max_steps  # type: ignore
    sirs[0] = np.copy(sir0)
    D = np.copy(M)
    N = M.shape[0]
    vis_func = Visualize(layout) if layout is not None else None
    if vis_func is not None:
        vis_func(nx.Graph(D), sirs[0])

    for step in range(1, max_steps):
        # Get the adjacency matrix to use at this step
        D = update_connections(D, M, step, sirs[step-1])

        # next_sir is the workhorse of the simulation because it is responsible
        # for simulating the disease spread
        sirs[step], states_changed = next_sir(sirs[step-1], D, disease, rand)
        if vis_func is not None:
            vis_func(nx.Graph(D), sirs[step])

        # find all the agents that are in the removed state. If that number is N,
        # the simulation is done.
        all_nodes_infected = len(np.where(sirs[step][2] > 0)[0]) == N
        if (not states_changed) and all_nodes_infected:
            return sirs[:step]

        # If there aren't any exposed or infectious agents, the disease is gone and we
        # can take a short cut to finish the simulation.
        disease_gone = np.sum(sirs[step][1]) == 0
        if (not states_changed) and disease_gone:
            for i in range(step, max_steps):
                sirs[i] = np.copy(sirs[step])
            return sirs

    return sirs


@njit
def next_sir(old_sir: np.ndarray, M: np.ndarray, disease: Disease, rand) -> Tuple[np.ndarray, bool]:
    """
    Use the disease to make the next SIR matrix also returns whether or not the old one differs from
    the new. The first dimension of sir is state. The second dimension is node.
    """

    sir = np.copy(old_sir)
    N = M.shape[0]
    probs = rand.random(N)

    # infectious to recovered
    to_r_filter = sir[1] > disease.days_infectious
    sir[2, to_r_filter] = -1
    sir[1, to_r_filter] = 0

    # susceptible to infectious
    i_filter = sir[1] > 0
    to_i_probs = 1 - np.prod(1 - (M * disease.trans_prob)[i_filter], axis=0)
    to_i_filter = (sir[0] > 0) & (probs < to_i_probs)
    sir[1, to_i_filter] = -1
    sir[0, to_i_filter] = 0

    sir[sir > 0] += 1
    sir[sir < 0] = 1

    return sir, to_r_filter.any() or to_i_filter.any()


def no_update(D: np.ndarray, M: np.ndarray, time_step: int, sir: np.ndarray) -> np.ndarray:
    """Dynamic function that actually isn't dynamic."""
    return D


def remove_dead_agents(D: np.ndarray, M: np.ndarray, time_step: int, sir: np.ndarray) -> np.ndarray:
    """Dynamic function that removes edges from agents in the R state."""
    new_D = np.copy(D)
    r_nodes = sir[2] > 0
    new_D[r_nodes, :] = 0
    new_D[:, r_nodes] = 0
    return new_D


class Visualize:
    """Show the network for .2 seconds."""
    def __init__(self, layout: Layout) -> None:
        """
        layout: Layout to use. This will not be automatically computed.
        """
        self._layout = layout
        self._state_to_color = {0: 'blue', 1: 'green', 2: 'grey'}

    def __call__(self, G: nx.Graph, sir: np.ndarray) -> None:
        node_colors = np.empty(len(G), dtype=np.object_)
        for state in range(sir.shape[0]):
            for node in range(sir.shape[1]):
                if sir[state, node] > 0:
                    node_colors[node] = self._state_to_color[state]
        plt.clf()
        nx.draw_networkx(G, pos=self._layout, with_labels=False,
                         node_color=node_colors, node_size=50)
        plt.pause(.2)  # type: ignore


def make_starting_sir(N: int, to_infect: Union[int, Tuple[int, ...]]) -> np.ndarray:
    """
    Make an initial SIR.

    N: number of agents in the simulation.
    to_infect: A tuple of agent id's or the number of agents to infect.
               If it is just a number, the agents will be randomly selected.
    """
    if isinstance(to_infect, int):
        to_infect = RAND.choice(N, size=to_infect)
    sir0 = np.zeros((3, N), dtype=np.int64)
    sir0[0] = 1
    sir0[1, to_infect] = 1
    sir0[0, to_infect] = 0
    return sir0


class FlickerBehavior:
    def __init__(self, M: np.ndarray,
                 edges_to_flicker: Collection[Tuple[int, int]],
                 flicker_pattern: Tuple[bool, ...],
                 name: Optional[str] = None) -> None:
        """
        Flickers inter-community edges according to flicker_pattern.

        G: The original network
        n_labels: How many labels to use for the label propogation algorithm that
                  finds inter-community edges
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


class FlickerConfig:
    def __init__(self, flicker_pattern: Tuple[bool, ...], name: str):
        self._flicker_pattern = flicker_pattern
        self._name = name

    def make_flicker_behavior(self, M: np.ndarray,
                              edges_to_flicker: Collection[Tuple[int, int]]) -> FlickerBehavior:
        return FlickerBehavior(M, edges_to_flicker, self._flicker_pattern, self._name)


def run_experiments(args: Tuple[str, int, int, Disease, Sequence[FlickerConfig]])\
        -> Optional[ExperimentResults]:
    """
    Run a batch of experiments and return a tuple containing the network's name,
    number of flickering edges, and a mapping of behavior name to the final
    amount of susceptible nodes. Return None on failure.

    args: (path to the network,
           number of sims to run for each behavior,
           simulation length,
           disease,
           a sequence of configs for the flickers to use.)
    """
    network_path, num_sims, sim_len, disease, flicker_configs = args
    G, layout, communities = read_network(network_path)
    if layout is None:
        print(f'{get_network_name(network_path)} has no layout.')
        return None
    if communities is None:
        print(f'{get_network_name(network_path)} has no community data.')
        return None
    M = nx.to_numpy_array(G)
    intercommunity_edges = {(u, v) for u, v in G.edges if communities[u] != communities[v]}
    N = M.shape[0]

    behavior_to_results: Dict[str, Sequence[int]] = {}
    for config in flicker_configs:
        behavior = config.make_flicker_behavior(M, intercommunity_edges)
        # The tuple comprehension is pretty arcane, so here is an explanation.
        # Each entry is the sum of the number of entries in the final SIR where
        # the days in S are greater than 0. That is to say, the number of
        # susceptible agents at the end of the simulation.
        num_sus = tuple(np.sum(simulate(M,
                                        make_starting_sir(N, 1),
                                        disease,
                                        behavior,
                                        sim_len,
                                        None)[-1][0] > 0)
                        for _ in range(num_sims))
        behavior_to_results[behavior.name] = num_sus

    return ExperimentResults(get_network_name(network_path), num_sims, sim_len,
                             len(intercommunity_edges)/len(G.edges), behavior_to_results)


if __name__ == '__main__':
    main()
