import sys
sys.path.append('')
from typing import Tuple
from sim_dynamic import make_starting_sir, simulate, Disease
from behavior import (AllPressureHandler, NoMitigation, DistancePressureHandler,
                      FlickerPressureBehavior, UpdateConnections)
from network import Network
import fileio as fio
import numpy as np
from numpy.random import Generator


def run_sims_on_class(class_name: str, sims_per_config: int, i0: int, disease: Disease,
                      rng: np.random.Generator):
    nets = fio.read_network_class(class_name)
    for net in nets:
        for intervention_strategy, name in (make_global_flicker(net, rng),
                                            make_local_flicker(net, rng),
                                            totally_isolate_inf_agents(net, rng),
                                            partially_isolate_inf_agents(net, rng)):
            results = []
            for _ in range(sims_per_config):
                sir0 = make_starting_sir(net.N, i0, rng)
                result = simulate(net.M, sir0, disease, intervention_strategy, 100, rng)
                results.append(result)
            fio.save_sim_results(f'{class_name} {name} (i0={i0})', results)


# TODO: verify all of these with Michael
# TODO: Decide what to do with hypothesis 3
def make_global_flicker(net: Network, rng: Generator) -> Tuple[UpdateConnections, str]:
    """For hypothesis 1"""
    return FlickerPressureBehavior(rng, AllPressureHandler()), 'Global Flicker'


def make_local_flicker(net: Network, rng: Generator) -> Tuple[UpdateConnections, str]:
    """For hypothesis 1"""
    return FlickerPressureBehavior(rng, DistancePressureHandler(net.dm, 1), .5), 'Local Flicker'


def totally_isolate_inf_agents(net: Network, rng: Generator) -> Tuple[UpdateConnections, str]:
    """For hypothesis 2"""
    return FlickerPressureBehavior(rng, DistancePressureHandler(net.dm, 0), 1),\
        'Totally Isolate Infected Agents'


def partially_isolate_inf_agents(net: Network, rng: Generator) -> Tuple[UpdateConnections, str]:
    """For hypothesis 2"""
    return FlickerPressureBehavior(rng, DistancePressureHandler(net.dm, 0), .5),\
        'Partially Isolate Infected Agents'
