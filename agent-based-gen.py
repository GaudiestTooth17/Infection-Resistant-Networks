#!/usr/bin/python3
from typing import Iterable, List, Callable, Sequence

from customtypes import Layout
import networkx as nx
import numpy as np
import sys
import matplotlib.pyplot as plt
from random import choice
from fileio import read_network_file
from analyzer import COLORS, calc_prop_common_neighbors


def main():
    # if len(sys.argv) < 2:
    #     print(f'Usage: {sys.argv[0]} <network>')
    #     return

    # M, layout = read_network_file(sys.argv[1])
    # G = nx.Graph(M)
    # N = len(G.nodes)
    N = 100
    G = nx.empty_graph(N)
    layout = None
    if layout is None:
        layout = nx.kamada_kawai_layout(G)

    # step = make_two_type_step(set(range(len(G.nodes)//10)), set(range(len(G.nodes)//10, len(G.nodes))))
    # step = homogenous_step
    step = make_time_based_step(N)
    for i in range(100):
        if i % 10 == 0:
            layout = nx.kamada_kawai_layout(G)
        plt.clf()
        plt.title(f'Step {i} |Components| == {len(tuple(nx.connected_components(G)))}')
        nx.draw_networkx_nodes(G, pos=layout, node_size=100, node_color=assign_colors(G))
        nx.draw_networkx_edges(G, pos=layout)
        plt.pause(.25)  # type: ignore
        step(G, layout)

    input('Press "enter" to continue.')


def assign_colors(G: nx.Graph) -> List[str]:
    components = nx.connected_components(G)
    node_to_color = [(node, COLORS[i]) for i, component in enumerate(components)
                     for node in component]
    node_to_color.sort(key=lambda x: x[0])
    return [color for _, color in node_to_color]


def homogenous_step(G: nx.Graph, layout: Layout) -> None:
    happy_number = 10
    for agent in G.nodes:
        neighbors = tuple(nx.neighbors(G, agent))
        # connect to a new neighbor
        if len(neighbors) == 0:
            to_add = choice(tuple(G.nodes))
            connect_agents(G, layout, agent, to_add)
        elif len(neighbors) < happy_number:
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
        elif len(neighbors) > happy_number:
            neighbor_to_strength = {(neighbor, calc_prop_common_neighbors(G, agent, neighbor))
                                    for neighbor in neighbors}
            to_remove = min(neighbor_to_strength, key=lambda x: x[1])[0]
            G.remove_edge(agent, to_remove)


def make_two_type_step(bridge_agents: Iterable[int], normal_agents: Iterable[int])\
                       -> Callable[[nx.Graph, Layout], None]:
    """
    agent_roles should contain two entries: 'bridge', 'normal'. The iterables
    associated with these keys should union to form the set of all nodes in G.
    normal agents will try to cluster around other agents.
    bridge agents will try to connect themselves to a few different clusters.
    """
    def two_type_step(G: nx.Graph, layout: Layout) -> None:
        normal_lb = 2  # lower bound
        normal_ub = 10  # upper bound
        bridge_happy_number = 2

        # how a normal agent behaves
        for agent in normal_agents:
            neighbors = tuple(nx.neighbors(G, agent))
            # connect to a new neighbor
            if len(neighbors) < normal_lb:
                to_add = choice(tuple(G.nodes))
                connect_agents(G, layout, agent, to_add)
            elif len(neighbors) < normal_ub:
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
            elif len(neighbors) > normal_ub:
                neighbor_to_strength = {(neighbor, calc_prop_common_neighbors(G, agent, neighbor))
                                        for neighbor in neighbors}
                to_remove = min(neighbor_to_strength, key=lambda x: x[1])[0]
                G.remove_edge(agent, to_remove)

        # how a bridge agent behaves
        for agent in bridge_agents:
            neighbors = tuple(nx.neighbors(G, agent))
            # search for more connections
            if len(neighbors) < bridge_happy_number:
                choices = [a for a in G.nodes if (a not in bridge_agents) and (a not in neighbors)]
                to_add = choice(choices)
                connect_agents(G, layout, agent, to_add)
            # if the agent has enough connections, look for ones to prune
            else:
                # connections are invalid if they are to an agent that shares a common neighbor
                invalid_connections = [a for a in neighbors if calc_prop_common_neighbors(G, agent, a) > 0]
                if len(invalid_connections) == 0:
                    invalid_connections = neighbors
                to_remove = choice(invalid_connections)
                G.remove_edge(agent, to_remove)

    return two_type_step


def make_time_based_step(N: int):
    time_satisfied = np.zeros(N, dtype='int')
    lower_bound = 7
    upper_bound = 9
    satisfied_border = 10

    def unsatisfied_behavior(G: nx.Graph, layout: Layout, agent: int, neighbors: Sequence[int]):
        # add a neighbor if lonely
        if len(neighbors) < lower_bound:
            if len(neighbors) == 0:
                connect_agents(G, layout, agent, choice(tuple(G.nodes)))
            else:
                neighbor_to_strength = {(neighbor, calc_prop_common_neighbors(G, agent, neighbor))
                                        for neighbor in neighbors}
                closest_neighbor = max(neighbor_to_strength, key=lambda x: x[1])[0]
                neighbor_choices = tuple(set(G[closest_neighbor]) - {agent})
                to_add = choice(neighbor_choices if len(neighbor_choices) > 0 else tuple(G.nodes))
                connect_agents(G, layout, agent, to_add)
        # remove a neighbor if overwhelmed
        elif len(neighbors) > upper_bound:
            neighbor_to_strength = {(neighbor, calc_prop_common_neighbors(G, agent, neighbor))
                                    for neighbor in neighbors}
            farthest_neighbor = min(neighbor_to_strength, key=lambda x: x[1])[0]
            G.remove_edge(agent, farthest_neighbor)

    def satisfied_behavior(G: nx.Graph, layout: Layout, agent: int, neighbors: Sequence[int]):
        if len(neighbors) < upper_bound - 1:
            neigbhor_choices = [n for n in G.nodes
                                if (time_satisfied[n] > satisfied_border) and (n not in neighbors)]
            if len(neigbhor_choices) > 0:
                connect_agents(G, layout, agent, choice(neigbhor_choices))

    def time_based_step(G: nx.Graph, layout: Layout) -> None:
        for agent in G.nodes:
            neighbors = tuple(G[agent])
            # choose behavior
            if time_satisfied[agent] < satisfied_border:
                unsatisfied_behavior(G, layout, agent, neighbors)
            else:
                satisfied_behavior(G, layout, agent, neighbors)
            # update satisfaction
            if lower_bound <= len(neighbors) < upper_bound:
                time_satisfied[agent] += 1
            else:
                time_satisfied[agent] = 0

    return time_based_step


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
