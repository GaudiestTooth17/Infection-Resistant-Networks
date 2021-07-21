from typing import Callable, Collection, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from customtypes import Layout
RNG = np.random.default_rng()

UpdateConnections = Callable[[np.ndarray, np.ndarray, int, np.ndarray], np.ndarray]
"""
Update the dynamic matrix D.
Parameters are D, M, current step, current sir array
"""


@dataclass
class Disease:
    days_infectious: int
    trans_prob: float


def pool_friendly_simulate(args):
    M, n_to_infect, disease, behavior, max_steps = args
    sir0 = make_starting_sir(M.shape[0], n_to_infect)
    return simulate(M, sir0, disease, behavior, max_steps, None)[-1]


def simulate(M: np.ndarray,
             sir0: np.ndarray,
             disease: Disease,
             update_connections: UpdateConnections,
             max_steps: int,
             layout: Optional[Layout],
             rng=RNG) -> List[np.ndarray]:
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
        sirs[step], states_changed = next_sir(sirs[step-1], D, disease, rng)
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


def next_sir(old_sir: np.ndarray, M: np.ndarray, disease: Disease, rng) -> Tuple[np.ndarray, bool]:
    """
    Use the disease to make the next SIR matrix also returns whether or not the old one differs from
    the new. The first dimension of sir is state. The second dimension is node.
    """

    sir = np.copy(old_sir)
    N = M.shape[0]
    probs = rng.random(N)

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


def make_starting_sir(N: int, to_infect: Union[int, Tuple[int, ...]], rng=RNG) -> np.ndarray:
    """
    Make an initial SIR.

    N: number of agents in the simulation.
    to_infect: A tuple of agent id's or the number of agents to infect.
               If it is just a number, the agents will be randomly selected.
    """
    if isinstance(to_infect, int):
        to_infect = rng.choice(N, size=to_infect)
    sir0 = np.zeros((3, N), dtype=np.int64)
    sir0[0] = 1
    sir0[1, to_infect] = 1
    sir0[0, to_infect] = 0
    return sir0


class StaticFlickerBehavior:
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
                 name: Optional[str] = None,
                 rng=RNG) -> None:
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
