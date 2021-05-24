#!/usr/bin/python3
from typing import List

from customtypes import Layout
import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from random import choice
from fileio import read_network_file
from analyzer import COLORS, calc_prop_common_neighbors


def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <network>')
        return

    M, layout = read_network_file(sys.argv[1])
    G = nx.Graph(M)
    if layout is None:
        layout = nx.kamada_kawai_layout(G)

    for _ in range(100):
        plt.clf()
        nx.draw_networkx_nodes(G, pos=layout, node_size=100, node_color=assign_colors(G))
        nx.draw_networkx_edges(G, pos=layout)
        plt.pause(.25)  # type: ignore
        step(G, layout)
    input()


def assign_colors(G: nx.Graph) -> List[str]:
    components = nx.connected_components(G)
    node_to_color = [(node, COLORS[i]) for i, component in enumerate(components)
                     for node in component]
    node_to_color.sort(key=lambda x: x[0])
    return [color for _, color in node_to_color]


def step(G: nx.Graph, layout: Layout) -> None:
    for agent in G.nodes:
        neighbors = tuple(nx.neighbors(G, agent))
        # connect to a new neighbor
        if len(neighbors) == 0:
            to_add = choice(tuple(G.nodes))
            connect_agents(G, layout, agent, to_add)
        elif len(neighbors) < 3:
            neighbor_to_strength = {(neighbor, calc_prop_common_neighbors(G, agent, neighbor))
                                    for neighbor in neighbors}
            closest_neighbor = max(neighbor_to_strength, key=lambda x: x[1])[0]
            new_neighbor_choices = set(nx.neighbors(G, closest_neighbor)) - {agent}
            if len(new_neighbor_choices) > 0:
                to_add = choice(tuple(new_neighbor_choices))
            else:
                to_add = choice(tuple(G.nodes))
            connect_agents(G, layout, agent, to_add)
        # disconnect from a neighbor
        elif len(neighbors) > 7:
            neighbor_to_strength = {(neighbor, calc_prop_common_neighbors(G, agent, neighbor))
                                    for neighbor in neighbors}
            to_remove = min(neighbor_to_strength, key=lambda x: x[1])[0]
            G.remove_edge(agent, to_remove)


def connect_agents(G: nx.Graph, layout: Layout, u: int, v: int) -> None:
    """
    Connect agents u and v in the network G and update the layout.
    """
    # if u has no neighbors, move u close to v
    if len(G[u]) == 0:
        new_x, new_y = layout[v]
        new_x += choice(np.linspace(-.25, .25, 10))
        new_y += choice(np.linspace(-.25, .25, 10))
        layout[u] = (new_x, new_y)
    # alternatively, if u has neighbors and v doesn't, move v close to u
    elif len(G[v]) == 0:
        new_x, new_y = layout[u]
        new_x += choice(np.linspace(-.01, .01, 10))
        new_y += choice(np.linspace(-.01, .01, 10))
        layout[v] = (new_x, new_y)
    G.add_edge(u, v)


if __name__ == '__main__':
    try:
        main()
    except EOFError:
        print('Goodbye.')
    except KeyboardInterrupt:
        print('Goodbye')
