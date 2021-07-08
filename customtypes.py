from typing import List, Dict, Tuple, Union, Set, TypeVar, Generic, Sequence
from collections import namedtuple, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import wasserstein_distance


# color is a string and reach is a number
Agent = namedtuple('Agent', 'color reach')
NodeColors = Union[List[str], List[Tuple[int, int, int]]]
Layout = Dict[int, Tuple[float, float]]
Communities = Dict[int, int]
Number = Union[int, float]
T = TypeVar('T')


class CircularList(Generic[T]):
    def __init__(self, base_list: List[T]) -> None:
        self._list = base_list

    def __getitem__(self, i: Union[int, slice]) -> Union[T, List[T]]:
        if isinstance(i, int):
            return self._list[i % len(self._list)]
        return self._list[i]


class CommunityEdges:
    """
    Takes a node in brackets and returns all the outgoing edges of the community
    it belongs to.
    """
    def __init__(self, M: np.ndarray, layout: Layout,
                 sqrt_num_communities, days_to_quarantine) -> None:
        """
        sqrt_num_communities by sqrt_num_communities cells are created and
        nodes get placed in one of these
        :param M: Adjacency matrix
        :param layout: layout is used to partition the graph
        :param num_communities: The number of communities to split the graph into
        """
        # create communities based off location of nodes in layout
        divisions = np.linspace(-1, 1, sqrt_num_communities+1)

        def find_cell_index(x) -> int:
            for i, _ in enumerate(divisions[:-1]):
                if divisions[i] <= x <= divisions[i+1]:
                    return i
            raise Exception(f'Cannot find {x}')

        node_to_community = {node: find_cell_index(x)*sqrt_num_communities+find_cell_index(y)
                             for node, (x, y) in layout.items()}

        edges = zip(*np.where(M > 0))  # type: ignore
        community_to_outgoing_edges = defaultdict(lambda: set())
        for u, v in edges:  # type: ignore
            if node_to_community[u] != node_to_community[v]:
                community_to_outgoing_edges[node_to_community[u]].add((u, v))

        self._node_to_community: Dict[int, Tuple[int, int]] = node_to_community
        self._community_to_outgoing_edges = {community: outgoing_edges
                                             for community, outgoing_edges
                                             in community_to_outgoing_edges.items()}
        self._M = np.copy(M)
        self.c_quarantine = np.zeros(sqrt_num_communities**2)
        self.days_to_quarantine = days_to_quarantine

    def quarantine_community_by_id(self, communtity_id):
        for u, v in self._community_to_outgoing_edges[communtity_id]:
            self._M[u, v] = 0
            self._M[v, u] = 0

    def unquarantine_community_by_id(self, communtity_id):
        for u, v in self._community_to_outgoing_edges[communtity_id]:
            self._M[u, v] = 1
            self._M[v, u] = 1

    def quarantine_community(self, agents: np.ndarray) -> np.ndarray:
        """
        :param agent: indices of agent whose community will be quarantined
        """
        self.c_quarantine += (self.c_quarantine > 0)

        communities_to_unquarantine = np.where(self.c_quarantine > self.days_to_quarantine)[0]
        for c in communities_to_unquarantine:
            self.c_quarantine[c] = 0
            self.unquarantine_community_by_id(c)

        agents = np.where(agents)[0]
        communities_to_quarantine = {self._node_to_community[agent] for agent in agents}
        # print('q communities:', communities_to_quarantine)

        for c in communities_to_quarantine:
            if self.c_quarantine[c] == 0:
                self.c_quarantine[c] = 1
                self.quarantine_community_by_id(c)

        return self._M

    def get_community_outgoing_edges(self, agent: int) -> Set[Tuple[int, int]]:
        """
        Returns the outgoing edges of the community that agent belongs to.
        """
        return self._community_to_outgoing_edges[self._node_to_community[agent]]


class ExperimentResults:
    def __init__(self, network_name: str,
                 sims_per_behavior: int,
                 sim_len: int,
                 proportion_flickering_edges: float,
                 behavior_to_num_sus: Dict[str, Sequence[int]],
                 baseline_behavior: str):
        """
        network_name
        sims_per_behavior
        behavior_to_num_sus: How many agents were still susceptible at the end of
                            each simulation with the specified behavior.
        baseline_behavior: The name of the behavior to computer the
                        Wasserstein distance of the others against.
        """
        self.network_name = network_name
        self.sims_per_behavior = sims_per_behavior
        self.sim_len = sim_len
        self.proportion_flickering_edges = proportion_flickering_edges
        self.behavior_to_num_sus = behavior_to_num_sus
        # Fail early if an incorrect name is supplied.
        if baseline_behavior not in behavior_to_num_sus:
            print(f'{baseline_behavior} is not in {list(behavior_to_num_sus.keys())}.'
                  'Fix this before continuing.')
            exit(1)
        self.baseline_behavior = baseline_behavior

    def save(self, directory: str, with_histograms: bool = False) -> None:
        """Save a histogram and a text file with analysis information in directory."""
        path = os.path.join(directory, self.network_name)
        if not os.path.exists(path):
            os.mkdir(path)

        # File Heading
        file_lines = [f'Name: {self.network_name}\n',
                      f'Number of sims per behavior: {self.sims_per_behavior}\n',
                      f'Simulation length: {self.sim_len}\n'
                      f'Proportion of edges flickering: {self.proportion_flickering_edges:.4f}\n\n']
        baseline_distribution = self.behavior_to_num_sus[self.baseline_behavior]
        for behavior_name, results in self.behavior_to_num_sus.items():
            # possibly save histograms
            if with_histograms:
                plt.figure()
                title = f'{self.network_name} {behavior_name}\n'\
                    f'sims={self.sims_per_behavior} sim_len={self.sim_len}'
                plt.title(title)
                plt.xlabel('Number of Susceptible Agents')
                plt.ylabel('Frequency')
                plt.hist(results, bins=None)
                plt.savefig(os.path.join(path, title+'.png'), format='png')

            # create a text entry for each behavior
            file_lines += [f'{behavior_name}\n',
                           f'Min:{np.min(results) : >20}\n',
                           f'Max:{np.max(results) : >20}\n',
                           f'Median:{np.median(results) : >20}\n',
                           f'Mean:{np.mean(results) : >20.3f}\n',
                           f'EMD from {self.baseline_behavior}:'
                           f'{wasserstein_distance(results, baseline_distribution) : >20.3f}\n\n']

        # save text entries
        with open(os.path.join(path, f'Report on {self.network_name}.txt'), 'w') as file:
            file.writelines(file_lines)
