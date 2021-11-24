import sys
sys.path.append('')
from sim_dynamic import make_starting_sir, simulate, Disease
from behavior import (NoMitigation, DistancePressureHandler,
                      FlickerPressureBehavior, UpdateConnections)
from network import Network
import fileio as fio
import numpy as np
from numpy.random import Generator


def run_sims_on_class(class_name: str, sims_per_config: int, i0: int, disease: Disease,
                      rng: np.random.Generator):
    nets = fio.read_network_class(class_name)
    for net in nets:
        for behavior_class in (NoMitigation, FlickerPressureBehavior):
            results = []
            for _ in range(sims_per_config):
                sir0 = make_starting_sir(net.N, i0, rng)
                behavior = make_behavior(net, behavior_class.name, rng)
                result = simulate(net.M, sir0, disease, behavior, 100, rng)
                results.append(result)
            fio.save_sim_results(f'{class_name} {behavior_class} (i0={i0})', results)


def make_behavior(net: Network, behavior_name, rng: Generator) -> UpdateConnections:
    if behavior_name == NoMitigation.name:
        return NoMitigation()
    elif behavior_name == FlickerPressureBehavior.name:
        pressure_handler = DistancePressureHandler(net.dm, 2)
        return FlickerPressureBehavior(rng, pressure_handler)
    raise ValueError(f'Unknown behavior "{behavior_name}"')


class Box:
    def __init__(self, data) -> None:
        self.data = data
