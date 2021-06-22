from typing import Callable, List, Optional, Tuple
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from fileio import read_network_file
from customtypes import Layout
RAND = np.random.default_rng(seed=0)

UpdateConnections = Callable[[np.ndarray, np.ndarray, int, np.ndarray], np.ndarray]
"""
Update the dynamic matrix D.
Parameters are D, M, current step, current sir array
"""


class Disease:
    def __init__(self, days_infectious: int, trans_prob: float) -> None:
        self.days_infectious = days_infectious
        self.trans_prob = trans_prob


def simulate(M: np.ndarray,
             sir0: np.ndarray,
             disease: Disease,
             update_connections: UpdateConnections,
             max_steps: int,
             layout: Optional[Layout]) -> List[np.ndarray]:
    rand = RAND  # increase access speed by making a local reference
    sirs: List[np.ndarray] = [None] * max_steps  # type: ignore
    sirs[0] = np.copy(sir0)
    D = np.copy(M)
    N = M.shape[0]
    vis_func = Visualize(layout) if layout is not None else None

    for step in range(1, max_steps):
        if vis_func is not None:
            vis_func(nx.Graph(D), sirs[step-1])
        # Get the adjacency matrix to use at this step
        D = update_connections(D, M, step, sirs[step-1])

        # next_sir is the workhorse of the simulation because it is responsible
        # for simulating the disease spread
        sirs[step], states_changed = next_sir(sirs[step-1], D, disease, rand)

        # find all the agents that are in the removed state. If that number is N,
        # the simulation is done.
        all_nodes_infected = len(np.where(sirs[step][2] > 0)[0]) == N
        if (not states_changed) and all_nodes_infected:
            print('Ending early because all nodes are infected.')
            return sirs[:step]

        # If there aren't any exposed or infectious agents, the disease is gone and we
        # can take a short cut to finish the simulation.
        disease_gone = np.sum(sirs[step][1]) == 0
        if (not states_changed) and disease_gone:
            print('Ending early because disease is gone.')
            for i in range(step, max_steps):
                sirs[i] = np.copy(sirs[step])
            return sirs

    return sirs


def next_sir(old_sir: np.ndarray, M: np.ndarray, disease: Disease, rand) -> Tuple[np.ndarray, bool]:
    """
    Use the disease to make the next SIR matrix also returns whether or not the old one differs from
    the new. The first dimension of sir is state. The second dimension is node.
    """

    seir = np.copy(old_sir)
    N = M.shape[0]
    probs = rand.random(N)

    # infectious to recovered
    to_r_filter = seir[1] > disease.days_infectious
    seir[2, to_r_filter] = -1
    seir[1, to_r_filter] = 0

    # susceptible to infectious
    i_filter = seir[1] > 0
    to_i_probs = (1 - np.prod(1 - (M * disease.trans_prob)[:, i_filter], axis=1)).reshape((N,))
    to_i_filter = (seir[0] > 0) & (probs < to_i_probs)
    seir[1, to_i_filter] = -1
    seir[0, to_i_filter] = 0

    seir[seir > 0] += 1
    seir[seir < 0] = 1

    return seir, to_r_filter.any() or to_i_filter.any()


def no_update(D: np.ndarray, M: np.ndarray, time_step: int, sir: np.ndarray) -> np.ndarray:
    return D


def remove_dead_agents(D: np.ndarray, M: np.ndarray, time_step: int, sir: np.ndarray) -> np.ndarray:
    new_D = np.copy(D)
    r_nodes = sir[2] > 0
    new_D[r_nodes, :] = 0
    new_D[:, r_nodes] = 0
    return new_D


class Visualize:
    def __init__(self, layout: Layout) -> None:
        self._layout = layout
        self._state_to_color = {0: 'blue', 1: 'green', 2: 'grey'}

    def __call__(self, G: nx.Graph, sir: np.ndarray) -> None:
        node_to_state = [(node, 0) for node in sir[0, sir[0] > 0]]\
            + [(node, 1) for node in sir[1, sir[1] > 0]] + [(node, 2) for node in sir[2, sir[2] > 0]]  # type: ignore
        node_to_state.sort(key=lambda x: x[0])
        node_colors = tuple(self._state_to_color[ns[1]] for ns in node_to_state)
        plt.clf()
        nx.draw_networkx(G, pos=self._layout, with_labels=False, node_color=node_colors, node_size=50)
        # plt.pause(.2)  # type: ignore
        plt.show()


if __name__ == '__main__':
    M, layout = read_network_file('networks/annealed-medium-diameter.txt')
    disease = Disease(4, .25)
    sir0 = np.zeros((3, M.shape[0]), dtype=np.int64)
    sir0[0] = 1
    sir0[1, 0] = 1
    sir0[0, 0] = 0

    simulate(M, sir0, disease, remove_dead_agents, 100, layout)
