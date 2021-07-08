#!/usr/bin/python3
from typing import Iterable, List, Callable, Optional, Sequence, Tuple, Union
import networkx as nx
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import choice
from fileio import old_read_network_file
from analyzer import COLORS, calc_prop_common_neighbors
Behavior = Callable[[nx.Graph], Tuple[nx.Graph, bool]]


def make_agent_generated_network(starting_point: Union[int, nx.Graph], behavior: Behavior)\
        -> nx.Graph:
    if isinstance(starting_point, int):
        G: nx.Graph = nx.empty_graph(starting_point)
    else:
        G = starting_point
    for _ in range(150):
        G, finished = behavior(G)
        if finished:
            break
    return G


def assign_colors(G: nx.Graph) -> List[str]:
    components = nx.connected_components(G)
    node_to_color = [(node, COLORS[i]) for i, component in enumerate(components)
                     for node in component]
    node_to_color.sort(key=lambda x: x[0])
    return [color for _, color in node_to_color]  # type: ignore


def homogenous_step(G: nx.Graph) -> None:
    """
    All agents behave the same, and that behave doesn't vary with time.
    The agents are trying to reach happy_number connections. If they are not connected
    to any other agents, they choose a random one to connect to. If they are connected
    to less than happy_number, they connect to one of the agents connected to their
    neighbor with the most common neighbors. If they have too many connections, they
    disconnect from the neighbor with the fewest common neighbors.
    """
    happy_number = 10
    for agent in G.nodes:
        neighbors = tuple(nx.neighbors(G, agent))
        # connect to a new neighbor
        if len(neighbors) == 0:
            to_add = choice(tuple(G.nodes))
            connect_agents(G, agent, to_add)
        elif len(neighbors) < happy_number:
            neighbor_to_strength = {(neighbor, calc_prop_common_neighbors(G, agent, neighbor))
                                    for neighbor in neighbors}
            closest_neighbor = max(neighbor_to_strength, key=lambda x: x[1])[0]
            new_neighbor_choices = set(nx.neighbors(G, closest_neighbor)) - {agent}
            if len(new_neighbor_choices) > 0:
                to_add = choice(tuple(new_neighbor_choices))
            else:
                to_add = choice(tuple(G.nodes))
            connect_agents(G, agent, to_add)
        # disconnect from a neighbor
        elif len(neighbors) > happy_number:
            neighbor_to_strength = {(neighbor, calc_prop_common_neighbors(G, agent, neighbor))
                                    for neighbor in neighbors}
            to_remove = min(neighbor_to_strength, key=lambda x: x[1])[0]
            G.remove_edge(agent, to_remove)


def make_two_type_step(bridge_agents: Iterable[int], normal_agents: Iterable[int])\
                       -> Callable[[nx.Graph], None]:
    """
    agent_roles should contain two entries: 'bridge', 'normal'. The iterables
    associated with these keys should union to form the set of all nodes in G.
    normal agents will try to cluster around other agents.
    bridge agents will try to connect themselves to a few different clusters.
    """
    def two_type_step(G: nx.Graph) -> None:
        normal_lb = 2  # lower bound
        normal_ub = 10  # upper bound
        bridge_happy_number = 2

        # how a normal agent behaves
        for agent in normal_agents:
            neighbors = tuple(nx.neighbors(G, agent))
            # connect to a new neighbor
            if len(neighbors) < normal_lb:
                to_add = choice(tuple(G.nodes))
                connect_agents(G, agent, to_add)
            elif len(neighbors) < normal_ub:
                neighbor_to_strength = {(neighbor, calc_prop_common_neighbors(G, agent, neighbor))
                                        for neighbor in neighbors}
                closest_neighbor = max(neighbor_to_strength, key=lambda x: x[1])[0]
                new_neighbor_choices = set(nx.neighbors(G, closest_neighbor)) - {agent}
                if len(new_neighbor_choices) > 0:
                    to_add = choice(tuple(new_neighbor_choices))
                else:
                    to_add = choice(tuple(G.nodes))
                connect_agents(G, agent, to_add)
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
                connect_agents(G, agent, to_add)
            # if the agent has enough connections, look for ones to prune
            else:
                # connections are invalid if they are to an agent that shares a common neighbor
                invalid_connections = [a for a in neighbors
                                       if calc_prop_common_neighbors(G, agent, a) > 0]
                if len(invalid_connections) == 0:
                    invalid_connections = neighbors
                to_remove = choice(invalid_connections)
                G.remove_edge(agent, to_remove)

    return two_type_step


class TimeBasedBehavior:
    def __init__(self, N: int,
                 lower_bound: int,
                 upper_bound: int,
                 steps_to_stable: int,
                 rand):
        """
        Agents try to have between [lower_bound, upper_bound] connections. However,
        behavior changes once an agent's connections have not changed for steps_to_stable
        steps. Whether or not they have changed is determined after all agents have taken
        an action, so if A connects to B, but B disconnects from A, A's connections will
        have not changed.

        In the unstable state, agents add neighbors if beneath lower_bound and remove agents
        if above upper_bound. Agents add a random agent adjacent to the agent with the most
        common neighbors. If that agent has no other neighbors, or the original agent has no
        neighbors, an agent is chosen at random to connect to.

        In the stable state, agents only look for more connections if they have less than
        upper_bound-1 connections. This is to minimize the probability that they start
        pruning connections after adding a new one.

        N: number of agents the simulation has.
        rand: an instance of np.random.default_rng
        """
        self._time_stable = np.zeros(N, np.uint64)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._steps_to_stable = steps_to_stable
        self._agent_to_previous_neighbors = {n: set() for n in range(N)}
        self._steps_taken = 0
        self._rand = rand

    def _unstable_behavior(self, G: nx.Graph, agent: int, neighbors: Sequence[int]):
        # add a neighbor if lonely
        if len(neighbors) < self._lower_bound:
            if len(neighbors) == 0:
                connect_agents(G, agent, choice(tuple(G.nodes)))
            else:
                neighbor_to_strength = {(neighbor, calc_prop_common_neighbors(G, agent, neighbor))
                                        for neighbor in neighbors}
                closest_neighbor = max(neighbor_to_strength, key=lambda x: x[1])[0]
                # TODO: neighbor_choices will likely include agents already adjacent to agent.
                # These should be filtered out.
                neighbor_choices = tuple(set(G[closest_neighbor]) - {agent})
                to_add = choice(neighbor_choices if len(neighbor_choices) > 0 else tuple(G.nodes))
                connect_agents(G, agent, to_add)
        # remove a neighbor if overwhelmed
        elif len(neighbors) > self._upper_bound:
            neighbor_to_strength = {(neighbor, calc_prop_common_neighbors(G, agent, neighbor))
                                    for neighbor in neighbors}
            farthest_neighbor = min(neighbor_to_strength, key=lambda x: x[1])[0]
            G.remove_edge(agent, farthest_neighbor)

    def _stable_behavior(self, G: nx.Graph, agent: int, neighbors: Sequence[int]):
        if len(neighbors) < self._upper_bound - 1:
            neighbor_choices = [n for n in G.nodes
                                if all((self._time_stable[n] > self._steps_to_stable,
                                        n not in neighbors,
                                        len(G[n]) < self._upper_bound - 1))]
            if len(neighbor_choices) > 0:
                connect_agents(G, agent, choice(neighbor_choices))

    def __call__(self, G: nx.Graph) -> Tuple[nx.Graph, bool]:
        G = nx.Graph(G)
        agents = np.array(G.nodes)
        self._rand.shuffle(agents)
        for agent in agents:
            neighbors = tuple(G[agent])
            # choose behavior
            if self._time_stable[agent] < self._steps_to_stable:
                self._unstable_behavior(G, agent, neighbors)
            else:
                self._stable_behavior(G, agent, neighbors)

        # Update satisfaction. Agents are satisifed by having consistant neighbors
        for agent in agents:
            neighbors = set(G[agent])
            if self._agent_to_previous_neighbors[agent] == neighbors:
                self._time_stable[agent] += 1
            else:
                self._time_stable[agent] = 0
            self._agent_to_previous_neighbors[agent] = neighbors

        self._steps_taken += 1
        return G, (self._time_stable > 0).all()


def connect_agents(G: nx.Graph, u: int, v: int) -> None:
    """
    Connect agents u and v in the network G.
    """
    G.add_edge(u, v)


def int_or_none(string: str) -> Optional[int]:
    try:
        return int(string)
    except ValueError:
        return None


def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <network or number of agents>')
        return

    N = int_or_none(sys.argv[1])
    if N is None:
        M, layout = old_read_network_file(sys.argv[1])
        G = nx.Graph(M)
        N = len(G.nodes)
    else:
        G = nx.empty_graph(N)
        layout = None
    if layout is None:
        layout = nx.kamada_kawai_layout(G)

    step = homogenous_step
    # step = make_two_type_step(set(range(len(G.nodes)//10)),
    #                           set(range(len(G.nodes)//10, len(G.nodes))))
    # step = make_time_based_step(N)
    node_size = 200
    for i in tqdm(range(150)):
        if i % 10 == 0:
            layout = nx.kamada_kawai_layout(G)
        plt.clf()
        plt.title(f'Step {i} |Components| == {len(tuple(nx.connected_components(G)))}')
        nx.draw_networkx(G, pos=layout, node_size=node_size, node_color=assign_colors(G),
                         with_labels=False)
        plt.pause(.2)  # type: ignore
        node_size = step(G)
    #     step(G, N)
    #     if nx.is_connected(G):
    #         print(f'Finished after {i+1} steps.')
    #         break

    # if nx.is_connected(G):
    #     output_network(G, f'agent-generated-{N}')
    # else:
    #     print('Network was not connected!')
    # input('Press "enter" to continue.')


if __name__ == '__main__':
    try:
        main()
    except EOFError:
        print('Goodbye.')
    except KeyboardInterrupt:
        print('Goodbye')
