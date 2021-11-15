import sys
sys.path.append('')
from matplotlib import pyplot as plt
from collections import defaultdict
from typing import Tuple
from sympy.core.function import diff
from sympy.core.symbol import Symbol
import numpy as np
from network import Network
import fileio as fio
import networkx as nx
import itertools as it
from sympy import Expr


def make_affiliation_network(f_coeffs: np.ndarray, g_coeffs: np.ndarray,
                             N: int, M: int, rng: np.random.Generator) -> Network:
    # get initial counts for how many stubs each agent and group has
    agent_stubs: np.ndarray = rng.choice(len(f_coeffs), N, p=f_coeffs)
    group_stubs: np.ndarray = rng.choice(len(g_coeffs), M, p=g_coeffs)
    # balance the counts so that the groups can combine nicely
    # This doesn't seem to be super necessary because it isn't feasible
    # to balance all the time and the results look fine visually without it
    # TODO: check that the actual results from the function match the results
    #       predicted by the math
    timeout = 1_000
    pre_calculed_indices = rng.choice(N+M, size=timeout)
    while np.sum(agent_stubs) != np.sum(group_stubs) and timeout > 0:
        stub_count_to_replace = pre_calculed_indices[timeout-1]
        if stub_count_to_replace < N:
            agent_stubs[stub_count_to_replace] = rng.choice(len(f_coeffs), 1, p=f_coeffs)
        else:
            group_stubs[stub_count_to_replace-N] = rng.choice(len(g_coeffs), 1, p=g_coeffs)
        timeout -= 1

    group_membership = defaultdict(lambda: set())
    # Assign agents to groups
    # This assumes that agents can be assigned to the same group multiple times, but only
    # one edge will appear
    agents_with_stubs = np.nonzero(agent_stubs)[0]
    groups_with_stubs = np.nonzero(group_stubs)[0]
    while len(agents_with_stubs) > 0 and len(groups_with_stubs) > 0:
        agent = rng.choice(agents_with_stubs)
        group = rng.choice(groups_with_stubs)
        group_membership[group].add(agent)
        agent_stubs[agent] -= 1
        group_stubs[group] -= 1
        agents_with_stubs = np.nonzero(agent_stubs)[0]
        groups_with_stubs = np.nonzero(group_stubs)[0]

    # for agent_id, n_stubs in enumerate(agent_stubs):
    #     if len(np.nonzero(group_stubs)[0]) == 0:
    #         break
    #     for _ in range(n_stubs):
    #         possible_groups = np.nonzero(group_stubs)[0]
    #         if len(possible_groups) == 0:
    #             break
    #         group = rng.choice(possible_groups)
    #         group_membership[group].add(agent_id)
    #         group_stubs[group] -= 1

    edges = it.chain.from_iterable(it.combinations(agents, 2)
                                   for agents in group_membership.values())
    G: nx.Graph = nx.empty_graph(N)
    G.add_edges_from(edges)
    return Network(G)


def coeffs_to_expr(coeffs: np.ndarray, var: Symbol) -> Expr:
    e: Expr = 0  # type: ignore
    for ex, c in enumerate(coeffs):
        e += c*(var**ex)
    return e


def calc_clustering_and_edge_density(f_coeffs: np.ndarray, g_coeffs: np.ndarray,
                                     N: int, M: int) -> Tuple[float, float]:
    x = Symbol('x')
    f = coeffs_to_expr(f_coeffs, x)
    g = coeffs_to_expr(g_coeffs, x)
    g_p = diff(g, x)
    G = f.subs({x: (g_p/g_p.subs({x: 1}))})
    G_p = diff(G, x)
    G_pp = diff(G_p, x)
    g_ppp = diff(g_p, x, 2)

    clustering = (M*g_ppp.subs({x: 1})) / (N*G_pp.subs({x: 1}))
    z = G_p.subs({x: 1})
    edge_density = z/(N-1)
    return clustering, edge_density


def display_results_for(f_coeffs, g_coeffs, seed):
    rng = np.random.default_rng(seed)
    N = 1000
    M = 100
    net = make_affiliation_network(f_coeffs, g_coeffs, N, M, rng)
    actual_c = nx.average_clustering(net.G)
    actual_ed = net.edge_density
    expected_c, expected_ed = calc_clustering_and_edge_density(f_coeffs, g_coeffs, N, M)
    print(f'Actual clustering: {actual_c}\nExpected clustering: {expected_c}\n')
    print(f'Actual edge density: {actual_ed}\nExpected edge density: {expected_ed}\n')


def ba_edge_density(N: int, m: int):
    """
    Calculate the edge density of the class of Barabasi-Albert networks with N nodes and m=m.
    """
    return (2*m*(N-m))/(N**2-N)


def make_barabasi_albert(N: int, approximate_edge_density: float) -> Network:
    """
    Return a Barabasi-Albert network with N nodes and some approximate edge density.

    Preconditions:
        0 < approximate_edge_density < .5
    """
    m = 1
    edge_density = ba_edge_density(N, m)
    ed_diff = np.abs(edge_density - approximate_edge_density)
    did_improve = True
    while did_improve:
        prev = (ed_diff, m)
        m += 1
        edge_density = ba_edge_density(N, m)
        ed_diff = np.abs(edge_density - approximate_edge_density)
        did_improve = ed_diff < prev[0]
    print(f'm = {prev[1]}')
    return Network(nx.barabasi_albert_graph(N, prev[1]))  # type: ignore


def ws_clustering(k: int, p: float):
    """
    Estimate the clustering coefficient of a Watts-Strogatz network
    using the method described in section III C of
    https://arxiv.org/pdf/cond-mat/9903411.pdf
    """
    return (3*(k-1))/(2*(2*k-1))*((1-p)**3)


def ws_edge_density(N: int, k: int):
    """
    Calculate the edge density of a Watts-Strogatz network
    """
    # Before rewiring, each node has k stubs.
    # It takes 2 stubs to make 1 edge.
    # Thus, there are Nk/2 edges
    # Unsimplified: (N*k/2) / ((N**2 - N)/2)
    # Simplified:
    return k/(N-1)


def test_affiliation_networks():
    seed = 666
    rng = np.random.default_rng(seed)
    for _ in range(10):
        f = rng.random(20)
        f /= np.sum(f)
        g = rng.random(100)
        g /= np.sum(g)
        display_results_for(f, g, seed)


def test_ws_clustering():
    N = 1000
    k = 8
    ps = np.linspace(0, 1, 20)
    n_trials = 40
    p_to_Gs = {p: [nx.connected_watts_strogatz_graph(N, k, p)
                   for _ in range(n_trials)]
               for p in ps}
    expected_clusterings = []
    actual_clusterings = []
    for p, Gs in p_to_Gs.items():
        expected_clusterings.append(ws_clustering(k, p))
        actual_clusterings.append([nx.average_clustering(G) for G in Gs])

    quartiles = np.quantile(actual_clusterings, (.25, .75), axis=1, interpolation='midpoint')
    mean_actual = np.mean(actual_clusterings, axis=1)
    name = f'Comparison of Clustering Estimation vs Reality\nk={k}'
    plt.title(name)
    plt.plot(ps, expected_clusterings)
    plt.plot(ps, mean_actual)
    plt.fill_between(ps, quartiles[0], quartiles[1], alpha=.4, color='orange')
    plt.legend(('Expected', f'Actual (Avg over {n_trials} trials)'))
    plt.xlabel('p')
    plt.ylabel('Clustering')
    plt.savefig(name.replace('\n', ' ')+'.png', format='png')


def test_ws_edge_density():
    N = 1000
    for k in range(2, 203, 2):
        G = nx.watts_strogatz_graph(N, k, 0)
        if len(G) == 0:
            print('no nodes. k =', k)
            continue
        net = Network(G)
        expected_edge_density = ws_edge_density(N, k)
        if net.edge_density != expected_edge_density:
            print(f'Expected: {expected_edge_density} Got: {net.edge_density}')


def gather_information_on_real_social_networks():
    secs_to_form_edge = 5*60  # 5 minutes
    net_paths = ('networks/sp_data_school_day_1_g.gexf',
                 'networks/sp_data_school_day_2_g.gexf',
                 'networks/High-School_data_2013.sp',
                 'networks/workplace_1st_deployment.sp',
                 'networks/workplace_2nd_deployment.sp')
    nets = [fio.read_socio_patterns_network(path, secs_to_form_edge)
            for path in net_paths]
    net_names = map(fio.get_network_name, net_paths)

    print(f'Networks were formed with {secs_to_form_edge} seconds to form an edge')
    for net, name in zip(nets, net_names):
        cc = nx.average_clustering(net.G)
        print(name)
        print(f'N: {net.N}')
        print(f'Clustering: {cc}')
        print(f'Edge Density = {net.edge_density}\n')


if __name__ == '__main__':
    gather_information_on_real_social_networks()
