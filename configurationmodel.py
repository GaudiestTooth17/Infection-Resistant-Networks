#!/usr/bin/python3
from typing import Callable, Sequence
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sa import make_sa_optimizer, make_fast_schedule


def main():
    n_steps = 500
    n_vertices = 100
    degrees = np.zeros(n_vertices, dtype='uint') + 4
    optimizer_step = make_sa_optimizer(clustering_objective, make_fast_schedule(.01),
                                       configuration_neighbor, degrees)
    pbar = tqdm(range(n_steps))
    energies = np.zeros(n_steps)
    for step in pbar:
        degrees, energy = optimizer_step()
        energies[step] = energy
        pbar.set_description('Energy: {:.5f}'.format(energy))

    plt.plot(energies)
    plt.show(block=False)
    plt.figure()
    plt.hist(degrees, bins=max(degrees) - min(degrees))
    plt.show(block=False)
    for _ in range(5):
        G = network_from_degree_sequence(degrees, make_vis_func(False), False)
        plt.figure()
        nx.draw_kamada_kawai(G)
        plt.show(block=False)

    input('Press <enter> to exit.')


def clustering_objective(degrees: Sequence[int]) -> float:
    networks = (network_from_degree_sequence(degrees, make_vis_func(False), False)
                for _ in range(25))
    clustering_coefficients = tuple(nx.average_clustering(G) for G in networks)
    return -sum(clustering_coefficients) / len(clustering_coefficients)  # type: ignore


def configuration_neighbor(degrees: Sequence[int]) -> Sequence[int]:
    neighbor = np.copy(degrees)
    nonzero_entries = np.where(neighbor > 0)[0]
    i, j = np.random.choice(nonzero_entries, 2, replace=False)
    neighbor[i] += np.random.choice((-1, 1))
    neighbor[j] += np.random.choice((-1, 1))
    return neighbor


def network_from_degree_sequence(degrees: Sequence[int], vis_func, force_good_behavior: bool) -> nx.Graph:
    if sum(degrees) % 2 != 0:
        raise Exception('The sum of degrees must be even.')

    node_to_remaining_stubs = dict(enumerate(degrees))
    G = nx.empty_graph(len(degrees))
    while sum(node_to_remaining_stubs.values()) > 0:
        available_nodes = tuple(filter(lambda u: node_to_remaining_stubs[u],
                                       node_to_remaining_stubs.keys()))
        u = np.random.choice(available_nodes)
        node_to_remaining_stubs[u] -= 1
        available_nodes = tuple(filter(lambda v: all((node_to_remaining_stubs[v],
                                                      any((not force_good_behavior,
                                                           v != u and not G.has_edge(u, v))))),
                                       node_to_remaining_stubs.keys()))
        if len(available_nodes) == 0:
            print('Network generation failed. Restarting.')
            return network_from_degree_sequence(degrees, vis_func, force_good_behavior)

        v = np.random.choice(available_nodes)
        node_to_remaining_stubs[v] -= 1
        G.add_edge(u, v)
        vis_func(G, u, v)

    return G


def make_vis_func(visualize: bool) -> Callable[[nx.Graph, int, int], None]:
    layout = None

    def do_vis(G: nx.Graph, u: int, v: int) -> None:
        nonlocal layout
        if layout is None:
            layout = nx.kamada_kawai_layout(G)
        nx.draw_networkx(G, with_labels=False, pos=layout,
                         node_color=tuple('green' if n in (u, v) else 'blue' for n in G.nodes))
        plt.pause(.5)  # type: ignore
        plt.clf()

    def dont_vis(G: nx.Graph, u: int, v: int) -> None:
        return None

    return do_vis if visualize else dont_vis


if __name__ == '__main__':
    try:
        main()
    except EOFError:
        print('\nGoodbye.')
    except KeyboardInterrupt:
        print('\nGoodbye.')
