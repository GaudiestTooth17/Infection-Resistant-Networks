from dataclasses import dataclass
from typing import Callable, Collection, List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.distance_measures import diameter
import numpy as np
from customtypes import Layout
from network import Network
from socialgood import get_distance_matrix
import behavior


@dataclass
class Disease:
    days_infectious: int
    trans_prob: float


def simulate(M: np.ndarray,
             sir0: np.ndarray,
             disease: Disease,
             update_connections: behavior.UpdateConnections,
             max_steps: int,
             rng,
             layout: Optional[Layout] = None) -> 'SimResults':
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
    sirs: List[np.ndarray] = [np.copy(sir0)]
    D = np.copy(M)
    N = M.shape[0]
    vis_func = Visualize(layout) if layout is not None else None
    if vis_func is not None:
        vis_func(nx.Graph(D), sirs[0], 0)

    # Needed data
    num_edges_removed = []
    current_edge_removal_durations = np.zeros(D.shape)
    total_edge_removal_durations = []
    num_pressured_nodes_at_step: List[int] = []
    diameter_at_step = []
    num_comps_at_step = []
    avg_comp_size_at_step = []
    last_perc_edges_removed_at_step = []

    for step in range(1, max_steps):
        # Get the adjacency matrix to use at this step
        D = update_connections(D, M, step, sirs[step - 1])

        # Gather the needed data
        num_edges_removed.append(update_connections.last_num_removed_edges)
        current_removed_edges = update_connections.last_removed_edges
        num_pressured_nodes_at_step.append(np.sum(update_connections.last_pressured_nodes))
        diameter_at_step.append(update_connections.last_diameter)
        num_comps_at_step.append(update_connections.last_num_comps)
        avg_comp_size_at_step.append(update_connections.last_avg_comp_size)
        last_perc_edges_removed_at_step.append(update_connections.last_perc_edges_removed)

        # Keeps only the currently_removed_edges, then adds one to each
        # old[np.where((new_rmvd == 0) * (old != 0))]
        # old -> current_edge_removal durations; new_rmvd -> current_removed_edges
        total_edge_removal_durations.extend(
            current_edge_removal_durations[(current_edge_removal_durations != 0)
                                           * (current_removed_edges == 0)]
        )
        current_edge_removal_durations = (current_edge_removal_durations * current_removed_edges)\
            + current_removed_edges

        # next_sir is the workhorse of the simulation because it is responsible
        # for simulating the disease spread
        sir, states_changed = next_sir(sirs[step - 1], D, disease, rng)
        sirs.append(sir)
        if vis_func is not None:
            vis_func(nx.Graph(D), sirs[step], step)

        # If there aren't any infectious agents, the disease is gone
        # and the simulation is done.
        disease_gone = np.sum(sirs[step][1]) == 0
        if (not states_changed) and disease_gone:
            break

    return SimResults(sirs, np.array(num_edges_removed), np.array(total_edge_removal_durations),
                      np.array(num_pressured_nodes_at_step), np.array(diameter_at_step),
                      np.array(num_comps_at_step), np.array(avg_comp_size_at_step),
                      last_perc_edges_removed_at_step)


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
                 sirs: Sequence[np.ndarray],
                 num_edges_removed_per_step: np.ndarray,
                 edge_removal_durations: np.ndarray,
                 pressured_nodes_at_step: np.ndarray,
                 diameter_at_step: np.ndarray,
                 num_comps_at_step: np.ndarray,
                 avg_comp_size_at_step: np.ndarray,
                 percent_edges_node_loses_at_step: Sequence[np.ndarray]
                 ):
        """
        Invasiveness
            Temporal average edges removed.
                Each step in behavior.last_num_removed_edges, aggregate in simulate
            Average edge removal duration in simulate using behavior.last_removed_edges
            Max number of edges removed at any given time
                during the simulation in simulate using behavior.last_num_removed_edges
        Isolation
            Diameter at each time step each step in behavior.last_diameter
            Number of components each step in behavior.last_num_comps
            Average component size each step in behavior.last_avg_comp_size
            % edges a node loses each step in behavior
        Survival
            survival rate: % of susceptible nodes at the end of the simulation
            Max number of infectious nodes at any given time during the simulation
        """
        self.num_steps = len(sirs)
        self.num_edges_removed_per_step = num_edges_removed_per_step
        self.edge_removal_durations = edge_removal_durations
        self.pressured_nodes_at_step = pressured_nodes_at_step
        self._diameter_at_step = diameter_at_step
        self._num_comps_at_step = num_comps_at_step
        self._avg_comp_size_at_step = avg_comp_size_at_step
        self._survival_rate = np.sum(sirs[-1][0] > 0) / sirs[-1].shape[1]
        self._max_num_infectious = max(np.sum(sir[1] > 0) for sir in sirs)
        self._percent_edges_node_loses_at_step = percent_edges_node_loses_at_step

    # Invasiveness

    @property
    def temporal_average_edges_removed(self) -> float:
        if self._temporal_average_edges_removed is None:
            self._temporal_average_edges_removed = np.average(self.num_edges_removed_per_step)
        return self._temporal_average_edges_removed

    @property
    def avg_edge_removal_duration(self) -> float:
        if self._avg_edge_removal_duration is None:
            self._avg_edge_removal_duration = np.average(self.edge_removal_durations)
        return self._avg_edge_removal_duration

    @property
    def max_num_edges_removed(self) -> int:
        if self._max_num_edges_removed is None:
            self._max_num_edges_removed = np.max(self.num_edges_removed_per_step)
        return self._max_num_edges_removed

    @property
    def avg_num_pressured_nodes(self) -> float:
        if self._avg_pressured_nodes is None:
            self._avg_pressured_nodes = np.average(self.pressured_nodes_at_step)
        return self._avg_pressured_nodes

    # Isolation

    @property
    def diameter_at_step(self) -> np.ndarray:
        return self._diameter_at_step

    @property
    def num_comps_at_step(self) -> np.ndarray:
        return self._num_comps_at_step

    @property
    def avg_comp_size_at_step(self) -> np.ndarray:
        return self._avg_comp_size_at_step

    @property
    def percent_edges_node_loses_at_step(self) -> Sequence[np.ndarray]:
        return self._percent_edges_node_loses_at_step

    # Surival

    @property
    def survival_rate(self) -> float:
        return self._survival_rate

    @property
    def max_num_infectious(self) -> int:
        """
        The max number of infectious agents over all time steps
        """
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
        plt.pause(.001)  # type: ignore


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
