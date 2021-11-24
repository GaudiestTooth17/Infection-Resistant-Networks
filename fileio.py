from network import Network
import os
import networkx as nx
import numpy as np
from typing import Counter, List, Optional, Sequence, Union, Callable, Tuple, Dict, Any
from customtypes import Layout, Communities, Number
import csv
import itertools as it
import copy
import os.path as op
import sys
from colorama import Fore, Style
import tarfile
import re
import matplotlib.pyplot as plt
import pickle
from simresults import SimResults
NETWORK_DIR = 'networks'


def write_network(G: nx.Graph,
                  network_name: str,
                  layout: Layout,
                  communities: Optional[Communities]) -> None:
    G = nx.Graph(G)
    # sometimes non tuple types slip through the cracks
    if not isinstance(layout[0], tuple):
        layout = {node: (location[0], location[1]) for node, location in layout.items()}

    nx.set_node_attributes(G, layout, 'layout')
    if communities is not None:
        nx.set_node_attributes(G, communities, 'community')

    nx.write_gml(G, network_name+'.txt')


def read_network(file_name: str,
                 remove_self_loops: bool = True)\
        -> Network:
    G: nx.Graph = nx.read_gml(file_name, None)  # type: ignore

    layout = nx.get_node_attributes(G, 'layout')
    if len(layout) != len(G):
        layout = None

    node_to_community = nx.get_node_attributes(G, 'community')
    if len(node_to_community) != len(G):
        node_to_community = None

    # get rid of the extraneous information
    G = nx.Graph(nx.to_numpy_array(G))
    if remove_self_loops:
        G.remove_edges_from(nx.selfloop_edges(G))

    if layout is not None:
        return Network(G, communities=node_to_community, layout=layout)
    return Network(G, communities=node_to_community)


# TODO: Instead of the cut off being for the longest stretch of time that two people
# are close to each other, maybe it should be for the total amount they are next to
# each other. This would match the elementary school data.
def read_socio_patterns_network(file_name: str,
                                seconds_to_form_edge: int) -> Network:
    """
    Read a network stored in the SocioPatterns format (http://www.sociopatterns.org/datasets/test/)

    file_name: Ends in '.dat' or '.sp'
    steps_to_form_edge: It is how many time steps (of 20s) two people need to spend
                        next to each other to have an edge between them in the
                        network
    """
    extension = op.splitext(file_name)[1]
    if extension == '.gexf':
        return _read_socio_patterns_gexf(file_name, seconds_to_form_edge)
    elif extension == '.sp':
        return _read_sociopatterns_sp(file_name, seconds_to_form_edge)
    raise ValueError(f'read_socio_patterns_network expected a .gexf or .sp file. Got: {file_name}')


def _read_socio_patterns_gexf(file_name: str, seconds_to_form_edge: int) -> Network:
    G: nx.Graph = nx.read_gexf(file_name)
    edges_to_remove = filter(lambda ea: ea[1]['duration'] < seconds_to_form_edge,
                             G.edges.items())
    G.remove_edges_from(edge[0] for edge in edges_to_remove)
    return Network(G)


def _read_sociopatterns_sp(file_name: str, seconds_to_form_edge: int) -> Network:
    with open(file_name, 'r') as f:
        lines = f.readlines()

    def line_to_time_and_edge(line: str) -> Tuple[int, Tuple[int, int]]:
        fields = line.split()
        if len(fields) < 3:
            raise Exception(f'Bad line: {line}')
        # There are some SocioPatterns networks that contain extra data on each line,
        # but the base format is always (time, u, v), I think.
        time_ = int(fields[0])
        edge = (int(fields[1]), int(fields[2]))
        # This is just in case the edges aren't always sorted the way the first
        # few appear to be. It will mess up the algorithm to have both
        # (u, v) and (v, u) present and this check makes sure that doesn't
        # happen.
        if edge[0] > edge[1]:
            edge = (edge[1], edge[0])
        return time_, edge

    step_to_edges = [line_to_time_and_edge(line) for line in lines]
    edge_to_time_active = Counter(x[1] for x in step_to_edges)

    # the magic number 20 is because each step is 20 seconds
    edges_to_keep = filter(lambda x: x[1]*20 >= seconds_to_form_edge, edge_to_time_active.items())
    G: nx.Graph = nx.empty_graph()
    G.add_edges_from(map(lambda x: x[0], edges_to_keep))

    return Network(G)


def old_output_network(G: nx.Graph, network_name: str,
                       layout_algorithm: Optional[Union[Callable, Layout]] = None):
    """
    Save the network to a file using the deprecated format.
    """
    with open(network_name+'.txt', 'w') as f:
        f.write(f'{len(G.nodes)}\n')

        # sometimes the nodes are identified by tuples instead of just integers
        # This doesn't work with other programs in the project, so we must give each tuple
        # a unique integer ID.
        node_to_id: Dict[Any, int] = {}
        current_id = 0

        def get_id_of_node(node) -> int:
            nonlocal current_id
            if node not in node_to_id:
                node_to_id[node] = current_id
                current_id += 1
            return node_to_id[node]

        edge_lines = [f'{get_id_of_node(n0)} {get_id_of_node(n1)}\n' for n0, n1 in G.edges]
        f.writelines(edge_lines)
        # this code is just for the visualization program I made ("graph-visualizer")
        # It writes out where each of the nodes should be drawn.
        f.write('\n')
        if layout_algorithm is None:
            layout_algorithm = nx.kamada_kawai_layout
        if callable(layout_algorithm):
            layout = layout_algorithm(G)
        else:
            layout = layout_algorithm
        f.writelines(f'{node_to_id[node]} {x} {y}\n'
                     for node, (x, y) in sorted(layout.items(), key=lambda x: x[0]))
        f.write('\n')


def old_read_network_file(file_name: str) -> Tuple[np.ndarray, Optional[Layout]]:
    """Read a network in the deprecated format."""
    with open(file_name, 'r') as f:
        line = f.readline()
        shape = (int(line[:-1]), int(line[:-1]))
        matrix = np.zeros(shape, dtype='uint8')

        line = f.readline()[:-1]
        i = 1
        while len(line) > 0:
            coord = line.split(' ')
            matrix[int(coord[0]), int(coord[1])] = 1
            matrix[int(coord[1]), int(coord[0])] = 1
            line = f.readline()[:-1]
            i += 1

        rest_of_lines = tuple(map(lambda s: s.split(),
                              filter(lambda s: len(s) > 1, f.readlines())))
        layout = {int(line[0]): (float(line[1]), float(line[2]))
                  for line in rest_of_lines} if len(rest_of_lines) > 0 else None
    return matrix, layout


def get_network_name(path_string: str) -> str:
    """Return only the name portion of the path to a network."""
    return '.'.join(op.basename(path_string).split('.')[:-1])


def network_names_to_paths(network_names: Sequence[str]) -> Sequence[str]:
    """Return the paths to the named networks. Report and exit if they cannot be found."""
    network_paths = tuple(op.join(NETWORK_DIR, name+'.txt') for name in network_names)
    error_free = True
    for path in network_paths:
        if not os.path.exists(path):
            print(Fore.RED+path, 'does not exist.', file=sys.stderr)
            error_free = False

    if not error_free:
        print('\nFix errors before continuing.'+Style.RESET_ALL, file=sys.stderr)
        exit(1)

    return network_paths


def read_network_class(class_name: str) -> Tuple[Network, ...]:
    """
    Return all the saved instances of the specified network class.

    Files should be saved in the root of a gunzipped tarball.
    """
    archive_name = class_name+'.tar.gz'
    extraction_dir = os.path.join('/tmp', class_name)

    with tarfile.open(os.path.join('networks', archive_name)) as tar:
        tar.extractall('/tmp')

    allowed_names = re.compile(r'instance-\d+.txt')
    id_num = re.compile(r'\d+')
    sorted_files = sorted(filter(lambda fname: allowed_names.match(fname),
                                 os.listdir(extraction_dir)),
                          key=lambda fname: int(id_num.search(fname).group()))  # type: ignore

    paths = (path for path in (os.path.join(extraction_dir, name)
                               for name in sorted_files))
    return tuple(read_network(path) for path in paths)


def write_network_class(class_name: str, nets: Sequence[Network]) -> None:
    tmp_dir = f'/tmp/{class_name}'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    target = os.path.join('networks', class_name + '.tar.gz')
    if os.path.exists(target):
        raise Exception(f'{target} already exists. Delete and try again.')
    with tarfile.open(target, 'x:gz') as tar:
        for i, net in enumerate(nets):
            tmp_name = f'instance-{i}'
            path_to_net = os.path.join(tmp_dir, tmp_name)
            write_network(net.G, path_to_net, net.layout, None)
            tar.add(path_to_net+'.txt', tmp_name+'.txt')


def save_animation(net: Network, sirs: List[np.ndarray], output_name: str) -> None:
    """Save an animation of the the sirs on the network."""
    pass


def save_sim_results(name: str, results: SimResults):
    """
    Save results in 'results/' as name with .pickle appended
    """
    with open('results/'+name+'.pickle', 'wb') as pickle_file:
        pickle.dump(results, pickle_file)


def load_sim_results(name: str) -> SimResults:
    """
    Load the named file ending in .pickle from 'results/' and return it
    """
    with open('results/'+name+'.pickle', 'rb') as pickle_file:
        return pickle.load(pickle_file)


class RawDataCSV:
    def __init__(self, title: str, distributions: Dict[str, Sequence[Number]]):
        self.title = title
        self.distributions = distributions

    def save(self):
        with open(os.path.join('results', self.title+'.csv'), 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            rows = it.chain(*[([dist_title], dist_data)
                              for dist_title, dist_data in self.distributions.items()])
            writer.writerows(rows)
        return self

    def save_boxplots(self):
        for dist_title, dist in self.distributions.items():
            plt.figure()
            plt.title(self.title)
            plt.xlabel(dist_title)
            plt.boxplot(dist)
            plt.savefig(os.path.join('results', dist_title+'.png'), format='png')
        return self

    @staticmethod
    def load_from_file(file_name: str) -> 'RawDataCSV':
        with open(file_name, 'r', newline='') as csv_file:
            reader = csv.reader(csv_file)
        raw_rows = tuple(reader)
        title = raw_rows[0][0]

        distributions: Dict[str, Sequence[Number]] = {}
        for i in range(1, len(raw_rows), 2):
            dist_title = raw_rows[i][0]
            dist_data = RawDataCSV._str_list_to_number_list(raw_rows[i+1])
            distributions[dist_title] = dist_data

        return RawDataCSV(title, distributions)

    @staticmethod
    def _str_list_to_number_list(str_list: List[str]) -> Union[List[int], List[float]]:
        num_list = []
        are_floats = False
        for s in str_list:
            if not are_floats and '.' in s:
                num_list = list(map(float, num_list))
                are_floats = True
            if are_floats:
                num_list.append(float(s))
            else:
                num_list.append(int(s))
        return num_list

    @staticmethod
    def union(title: str, x: 'RawDataCSV', y: 'RawDataCSV') -> 'RawDataCSV':
        new_data = copy.deepcopy(x.distributions)
        new_data.update(y.distributions)
        return RawDataCSV(title, new_data)
