import hcmioptim as ho
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import copy
from genfuncs import identity, make_scaler, make_right_shift, differentiation, summation
NUM_TO_TRANSFORMATION = dict(enumerate((identity, make_scaler(2),  make_right_shift(1),
                                        differentiation, summation)))


def main():
    # desired_sequence = np.array((0, 1, 4, 9, 16, 25, 36))  # example from pdf p. 6
    desired_sequence = np.array((9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
    base_sequence = np.ones(desired_sequence.shape, dtype=desired_sequence.dtype)
    transforms = np.zeros(6, dtype=np.int64)
    optimizer = ho.sa.SAOptimizer(make_sequence_objective(desired_sequence, base_sequence),
                                  ho.sa.make_fast_schedule(1000),
                                  sequence_neighbor,
                                  transforms,
                                  False)

    energies = []
    answer = None
    pbar = tqdm(range(50_000))
    for _ in pbar:
        answer, energy = optimizer.step()
        pbar.set_description(f'Energy: {energy}')
        energies.append(energy)

    print('answer:', answer)
    print('sequence:', trans_inds_to_seq(answer, base_sequence))
    print('energy:', energies[-1])
    # plt.plot(energies)
    # plt.show()


def make_sequence_objective(desired_sequence, base_sequence):
    def objective(transformation_indices):
        nonlocal desired_sequence, base_sequence
        transformation_indices %= len(NUM_TO_TRANSFORMATION)

        actual_sequence = trans_inds_to_seq(transformation_indices, base_sequence)
        # differentiation and right shifting change the size of the sequence
        # this make sure the sequences are the same size when they are compared
        length_diff = actual_sequence.shape[0] - desired_sequence.shape[0]
        if length_diff > 0:
            desired_sequence = np.append(desired_sequence, [0 for _ in range(length_diff)])
        elif length_diff < 0:
            actual_sequence = np.append(actual_sequence, [0 for _ in range(-length_diff)])

        energy = np.sum(np.abs(desired_sequence - actual_sequence))
        return energy
    return objective


def trans_inds_to_seq(indices, base):
    s = base
    for i in indices:
        s = NUM_TO_TRANSFORMATION[i](s)
    return np.array(tuple(s))


def sequence_neighbor(sequence: np.ndarray) -> np.ndarray:
    neighbor = np.copy(sequence)
    ind = 0
    while neighbor[ind] == sequence[ind]:
        ind = np.random.randint(neighbor.shape[0])
        neighbor[ind] = np.random.choice(tuple(range(len(NUM_TO_TRANSFORMATION))))
    return neighbor


if __name__ == '__main__':
    main()
