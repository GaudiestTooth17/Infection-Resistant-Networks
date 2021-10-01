from dataclasses import dataclass
from typing import Callable, Collection, List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from customtypes import Layout
from network import Network
from socialgood import get_distance_matrix
import behavior


@dataclass
class Disease:
    days_infectious: int
    trans_prob: float


def pool_friendly_simulate(args):
    M, n_to_infect, disease, behavior, max_steps, rng = args
    sir0 = make_starting_sir(M.shape[0], n_to_infect, rng)
    return simulate(M, sir0, disease, behavior, max_steps, None)[-1]


def simulate(M: np.ndarray,
             sir0: np.ndarray,
             disease: Disease,
             update_connections: behavior.UpdateConnections,
             max_steps: int,
             rng,
             layout: Optional[Layout] = None) -> List[np.ndarray]:
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
        vis_func(nx.Graph(D), sirs[0], 0)

    # Needed data
    num_egdes_removed = []
    current_edge_removal_durations = np.zeros(D.shape)
    total_edge_removal_durations = []
    # old[np.where((new == 0) * (old != 0))]

    for step in range(1, max_steps):
        # Get the adjacency matrix to use at this step
        D = update_connections(D, M, step, sirs[step - 1])

        # Gather the needed data
        num_egdes_removed.append(update_connections.last_num_removed_edges)
        current_removed_edges = update_connections.last_removed_edges
        # Keeps only the currently_removed_edges, then adds one to each
        new_edge_removal_durations = (current_edge_removal_durations * current_removed_edges) + current_removed_edges
        total_edge_removal_durations.extend(current_edge_removal_durations[(current_edge_removal_durations != 0) * (current_removed_edges == 0)])

        # next_sir is the workhorse of the simulation because it is responsible
        # for simulating the disease spread
        sirs[step], states_changed = next_sir(sirs[step - 1], D, disease, rng)
        if vis_func is not None:
            vis_func(nx.Graph(D), sirs[step], step)

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
    # return SimResults(M, sirs, pressured_nodes, removed_nodes, len(pressured_nodes))


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


def remove_dead_agents(D: np.ndarray, M: np.ndarray, time_step: int, sir: np.ndarray) -> np.ndarray:
    """Dynamic function that removes edges from agents in the R state."""
    new_D = np.copy(D)
    r_nodes = sir[2] > 0
    new_D[r_nodes, :] = 0
    new_D[:, r_nodes] = 0
    return new_D


class SimResults:
    def __init__(self,
                 M: np.ndarray,
                 sir: List[np.ndarray],
                 num_steps: int,

                 ) -> None:
        """
        Invasiveness
            Temporal average edges removed.
                Each step in behavior.last_num_removed_edges, aggregate in simulate
            Average edge removal duration TODO: in simulate using behavior.last_removed_edges
            Max number of edges removed at any given time
                during the simulation TODO: in simulate using behavior.last_num_removed_edges
        Isolation
            Diameter at each time step TODO: each step in behavior.last_diameter
            Number of components TODO: each step in behavior.last_num_comps
            Average component size TODO: each step in behavior.last_avg_comp_size
            % edges a node loses TODO: each step in behavior
        Survival
            survival rate: % of susceptible nodes at the end of the simulation TODO:
            Max number of infectious nodes at any given time during the simulation TODO:
        """
        self.M = M
        self.sir = sir
        self.num_steps = num_steps
        self.num_edges_removed = None  # TODO:
        self.edge_removal_durations = None  # TODO:

        self._temporal_average_edges_removed = None
        self._avg_edge_removal_duration = None
        self._max_num_edges_removed = None
        self._avg_pressured_nodes = None
        self._diameter_at_step = None
        self._num_comps_at_step = None
        self._avg_comp_size_at_step = None
        self._percent_edges_node_loses_at_step = None
        self._survival_rate = None
        self._max_num_infectious = None

    # Invasiveness

    @property
    def temporal_average_edges_removed(self):
        if self._temporal_average_edges_removed is None:
            self._temporal_average_edges_removed = sum(self.num_edges_removed) / len(self.num_edges_removed)
        return self._temporal_average_edges_removed

    @property  # TODO:
    def avg_edge_removal_duration(self):
        if self._avg_edge_removal_duration is None:
            self._avg_edge_removal_duration = None  # TODO:
        return self._avg_edge_removal_duration

    @property
    def max_num_edges_removed(self):
        if self._max_num_edges_removed is None:
            self._max_num_edges_removed = max(self.num_edges_removed)
        return self._max_num_edges_removed

    @property
    def avg_pressured_nodes(self):
        if self._avg_pressured_nodes is None:
            self._avg_pressured_nodes = np.sum(self.pressured_nodes) / self.num_steps
        return self._avg_pressured_nodes

    # Isolation

    @property  # TODO:
    def diameter_at_step(self):
        if self._diameter_at_step is None:
            self._diameter_at_step = None  # TODO:
        return self._diameter_at_step

    @property  # TODO:
    def num_comps_at_step(self):
        if self._num_comps_at_step is None:
            self._num_comps_at_step = None  # TODO:
        return self._num_comps_at_step

    @property  # TODO:
    def avg_comp_size_at_step(self):
        if self._avg_comp_size_at_step is None:
            self._avg_comp_size_at_step = None  # TODO:
        return self._avg_comp_size_at_step

    @property  # TODO:
    def percent_edges_node_loses_at_step(self):
        if self._percent_edges_node_loses_at_step is None:
            self._percent_edges_node_loses_at_step = None  # TODO:
        return self._percent_edges_node_loses_at_step

    # Surival

    @property  # TODO:
    def survival_rate(self):
        if self._survival_rate is None:
            self._survival_rate = None  # TODO:
        return self._survival_rate

    @property  # TODO:
    def max_num_infectious(self):
        if self._max_num_infectious is None:
            self._max_num_infectious = None  # TODO:
        return self._max_num_infectious


class Visualize:
    def __init__(self, layout: Layout) -> None:
        """
        Show the network for .2 seconds.

        layout: Layout to use. This will not be automatically computed.
        """
        self._layout = layout
        self._state_to_color = {0: 'blue', 1: 'green', 2: 'grey'}

    def __call__(self, G: nx.Graph, sir: np.ndarray, step) -> None:
        node_colors = np.empty(len(G), dtype=np.object_)
        for state in range(sir.shape[0]):
            for node in range(sir.shape[1]):
                if sir[state, node] > 0:
                    node_colors[node] = self._state_to_color[state]
        plt.clf()
        nx.draw_networkx(G, pos=self._layout, with_labels=False,
                         node_color=node_colors, node_size=50)
        plt.title(f'Step: {step}, S: {np.sum(sir[0, :] > 0)},'
                  f'I: {np.sum(sir[1, :] > 0)}, R: {np.sum(sir[2, :] > 0)}')
        # plt.show()
        plt.pause(.5)  # type: ignore


def make_starting_sir(N: int, to_infect: Union[int, Tuple[int, ...]], rng) -> np.ndarray:
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
