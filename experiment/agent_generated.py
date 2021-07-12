from common import safe_run_trials
from networkgen import TimeBasedBehavior, AgentBehavior, make_agent_generated_network
import time
import numpy as np
from sim_dynamic import Disease, FlickerBehavior, simulate, make_starting_sir
from typing import Optional, Tuple, Any
import partitioning
import networkx as nx
from socialgood import rate_social_good


def agent_generated_entry_point(seed=1337):
    print('Running agent generated experiments')
    num_trials = 100
    rand = np.random.default_rng(seed)
    N = 500
    lb_connection = 4
    ub_connection = 6
    steps_to_stability = 20
    agent_behavior = TimeBasedBehavior(N, lb_connection, ub_connection,
                                       steps_to_stability, rand)
    disease = Disease(4, .2)

    safe_run_trials(f'Agentbased {N}-{lb_connection}-{ub_connection}-{steps_to_stability}',
                    run_agent_generated_trial, (disease, agent_behavior, N, rand), num_trials)


def run_agent_generated_trial(args: Tuple[Disease, AgentBehavior, int, Any])\
        -> Optional[Tuple[float, float, float]]:
    """
    args: (disease to use in the simulation,
           the behavior agents have when generating the network,
           the number of agents in the network,
           an instance of np.random.default_rng)
    """
    disease, agent_behavior, N, rand = args
    sim_len = 200
    sims_per_trial = 150
    G = make_agent_generated_network(N, agent_behavior)
    if G is None:
        return None

    to_flicker = partitioning.fluidc_partition(G, 50)
    proportion_flickering = len(to_flicker) / len(G.edges)
    social_good = rate_social_good(G)
    M = nx.to_numpy_array(G)

    network_behavior = FlickerBehavior(M, to_flicker, (True, False), "Probs don't change this")
    avg_sus = np.mean([np.sum(simulate(M, make_starting_sir(len(M), 1),
                                       disease, network_behavior, sim_len, None, rand)[-1][0] > 0)
                       for _ in range(sims_per_trial)]) / len(M)

    return proportion_flickering, avg_sus, social_good


if __name__ == '__main__':
    agent_generated_entry_point(2)
