import sys
sys.path.append('')
from typing import Tuple
from sim_dynamic import make_starting_sir, simulate, Disease
from behavior import (AllPressureHandler, NoMitigation, DistancePressureHandler,
                      FlickerPressureBehavior, UpdateConnections,
                      BetweenDistancePressureHandler, MultiPressureBehavior)
from network import Network
import fileio as fio
import numpy as np
from numpy.random import Generator
from collections import defaultdict
from tqdm import tqdm


def main():
    class_names = 'BarabasiAlbert', 'WattsStrogatz'
    for class_name in class_names:
        rng = np.random.default_rng(69)
        run_sims_on_class(class_name, 25, 5, Disease(4, .3), rng)


def run_sims_on_class(class_name: str, sims_per_config: int, i0: int, disease: Disease,
                      rng: np.random.Generator):
    nets = fio.read_network_class(class_name)
    results = defaultdict(lambda: [])
    for net in tqdm(nets, desc=f'Cycling through {class_name} networks'):
        for intervention_strategy, name in (global_flicker_half(net, rng),
                                            local_flicker_half(net, rng),
                                            global_flicker_quarter(net, rng),
                                            local_flicker_quarter(net, rng),
                                            totally_isolate_inf_agents(net, rng),
                                            partially_isolate_inf_agents(net, rng),
                                            proposed_optimal_mitigation(net, rng)):
            for _ in range(sims_per_config):
                sir0 = make_starting_sir(net.N, i0, rng)
                result = simulate(net.M, sir0, disease, intervention_strategy, 100, rng)
                results[name].append(result)

    for mitigation_name, data in results.items():
        fio.save_sim_results(f'{class_name} {mitigation_name} (i0={i0})', data)


def no_mitigation(net: Network, rng: Generator) -> Tuple[UpdateConnections, str]:
    """For hypothesis 1"""
    return NoMitigation(), 'No Mitigation'


def global_flicker_quarter(net: Network, rng: Generator) -> Tuple[UpdateConnections, str]:
    """For hypothesis 1"""
    return FlickerPressureBehavior(rng, AllPressureHandler(), .25), f'Global Flicker .25'


def global_flicker_half(net: Network, rng: Generator) -> Tuple[UpdateConnections, str]:
    """For hypothesis 1"""
    return FlickerPressureBehavior(rng, AllPressureHandler(), .5), f'Global Flicker .5'


def local_flicker_half(net: Network, rng: Generator) -> Tuple[UpdateConnections, str]:
    """For hypothesis 1"""
    return FlickerPressureBehavior(rng, DistancePressureHandler(net.dm, 1), .5), 'Local Flicker .5'


def local_flicker_quarter(net: Network, rng: Generator) -> Tuple[UpdateConnections, str]:
    """For hypothesis 1"""
    return FlickerPressureBehavior(rng, DistancePressureHandler(net.dm, 1), .25),\
        'Local Flicker .25'


def totally_isolate_inf_agents(net: Network, rng: Generator) -> Tuple[UpdateConnections, str]:
    """For hypothesis 2"""
    return FlickerPressureBehavior(rng, DistancePressureHandler(net.dm, 0), 1),\
        'Totally Isolate Infected Agents'


def partially_isolate_inf_agents(net: Network, rng: Generator) -> Tuple[UpdateConnections, str]:
    """For hypothesis 2"""
    return FlickerPressureBehavior(rng, DistancePressureHandler(net.dm, 0), .5),\
        'Partially Isolate Infected Agents'


def proposed_optimal_mitigation(net: Network, rng: Generator) -> Tuple[UpdateConnections, str]:
    """For hypothesis 3"""
    dm = (- net.common_neighbors_matrix + np.sum(net.M, axis=0)) * (net.common_neighbors_matrix > 0)

    # fdm => flattened distance matrix
    fdm = sorted(dm[dm > 0].flatten())
    dm_size = len(fdm)
    # get the first quartile value
    quarter_dm_value = fdm[dm_size // 4]
    # get median value
    half_dm_value = fdm[dm_size // 2]
    ph1 = BetweenDistancePressureHandler(dm, quarter_dm_value, half_dm_value)
    ph2 = BetweenDistancePressureHandler(dm, half_dm_value, np.inf)

    behaviors = [
        FlickerPressureBehavior(rng, ph1, .5),
        FlickerPressureBehavior(rng, ph2, 1)
    ]
    update_behavior = MultiPressureBehavior(rng, behaviors)
    return update_behavior, 'Proposed Optimal Mitigation'


if __name__ == '__main__':
    main()
