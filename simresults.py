from typing import Sequence
import numpy as np


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
        self._max_num_edges_removed = None
        self._temporal_average_edges_removed = None
        self._avg_edge_removal_duration = None

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
