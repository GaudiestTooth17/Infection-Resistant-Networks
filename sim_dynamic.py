from typing import Callable, List, Tuple
import numpy as np
from fileio import read_network_file


class Disease:
    def __init__(self, days_infectious: int, trans_prob: float) -> None:
        self.days_infectious = days_infectious
        self.trans_prob = trans_prob


def simulate(M: np.ndarray,
             sir0: np.ndarray,
             disease: Disease,
             update_connections: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
             max_steps: int) -> List[np.ndarray]:
    rand = np.random.default_rng()
    sirs: List[np.ndarray] = [None] * max_steps  # type: ignore
    sirs[0] = np.copy(sir0)
    D = np.copy(M)
    N = M.shape[0]

    for step in range(1, max_steps):
        # Get the adjacency matrix to use at this step
        D = update_connections(D, M, step)

        # next_sir is the workhorse of the simulation because it is responsible
        # for simulating the disease spread
        sirs[step], states_changed = next_sir(sirs[step-1], D, disease, rand)

        # find all the agents that are in the removed state. If that number is N,
        # the simulation is done.
        all_nodes_infected = len(np.where(sirs[step][:, 2] > 0)[0]) == N
        if (not states_changed) and all_nodes_infected:
            return sirs[:step]

        # If there aren't any exposed or infectious agents, the disease is gone and we
        # can take a short cut to finish the simulation.
        disease_gone = np.sum(sirs[step][:, 1]) == 0
        if (not states_changed) and disease_gone:
            for i in range(step, max_steps):
                sirs[i] = np.copy(sirs[step])
            return sirs

    return sirs


def next_sir(old_sir: np.ndarray, M: np.ndarray, disease: Disease, rand) -> Tuple[np.ndarray, bool]:
    """
    Use the disease to make the next SIR matrix also returns whether or not the old one differs from
    the new. The first dimension of sir is node. The second dimension is state.
    """

    seir = np.copy(old_sir)
    N = M.shape[0]
    probs = rand.random(N)

    # infectious to recovered
    to_r_filter = seir[:, 2] > disease.days_infectious
    seir[to_r_filter, 2] = -1
    seir[to_r_filter, 1] = 0

    # susceptible to infectious
    i_filter = seir[:, 1] > 0
    to_i_probs = (1 - np.prod(1 - (M * disease.trans_prob)[:, i_filter], axis=1)).reshape((N,))
    to_i_filter = (seir[:, 0] > 0) & (probs < to_i_probs)
    seir[to_i_filter, 1] = -1
    seir[to_i_filter, 0] = 0

    seir[seir > 0] += 1
    seir[seir < 0] = 1

    return seir, to_r_filter.any() or to_i_filter.any() or to_i_filter.any()


def no_update(D: np.ndarray, M: np.ndarray, time_step: int) -> np.ndarray:
    return D


if __name__ == '__main__':
    M, _ = read_network_file('networks/annealed-medium-diameter.txt')
    disease = Disease(4, .25)
    sir0 = np.zeros((M.shape[0], 3), dtype=np.int64)
    sir0[0, 1] = 1
    sir0[0, 0] = 0

    simulate(M, sir0, disease, no_update, 100)
