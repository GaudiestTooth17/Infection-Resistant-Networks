import networkgen
from experiment.common import safe_run_trials
from networkgen import TimeBasedBehavior, AgentBehavior, make_agent_generated_network
import time
import numpy as np
from sim_dynamic import Disease, FlickerBehavior, simulate, make_starting_sir
from typing import Tuple, Any
import partitioning
import networkx as nx


def agent_generated_entry_point():
    print('Running agent generated experiments')
    start_time = time.time()
    num_trials = 1000
    rand = np.random.default_rng(1337)
    N = 500
    lb_connection = 4
    ub_connection = 6
    steps_to_stability = 20
    agent_behavior = TimeBasedBehavior(N, lb_connection, ub_connection,
                                       steps_to_stability, rand)
    disease = Disease(4, .2)

    safe_run_trials(f'Agentbased {N}-{lb_connection}-{ub_connection}-{steps_to_stability}',
                    run_agent_generated_trial, (disease, agent_behavior, N, rand), num_trials)

    print(f'Finished experiments with agent generated networks ({time.time()-start_time} s).')


def run_agent_generated_trial(args: Tuple[Disease, AgentBehavior, int, Any]) -> Tuple[float, float]:
    """
    args: (disease to use in the simulation,
           the behavior agents have when generating the network,
           the number of agents in the network,
           an instance of np.random.default_rng)
    """
    disease, agent_behavior, N, rand = args
    sim_len = 200
    sims_per_trial = 150
    G = None
    while G is None:
        G = make_agent_generated_network(N, agent_behavior)

    to_flicker = partitioning.fluidc_partition(G, 50)
    proportion_flickering = len(to_flicker) / len(G.edges)
    M = nx.to_numpy_array(G)

    network_behavior = FlickerBehavior(M, to_flicker, (True, False), "Probs don't change this")
    avg_sus = np.mean([np.sum(simulate(M, make_starting_sir(len(M), 1),
                                       disease, network_behavior, sim_len, None, rand)[-1][0] > 0)
                       for _ in range(sims_per_trial)])

    return proportion_flickering, avg_sus
