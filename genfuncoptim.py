from typing import Tuple
import hcmioptim.ga as ga
import hcmioptim.sa as sa
from tqdm import tqdm
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from genfuncs import identity, make_scaler, make_right_shift, differentiation, summation
NUM_TO_TRANSFORMATION = dict(enumerate((identity, make_scaler(2),  make_right_shift(1),
                                        differentiation, summation)))


def main():
    with_ga()


def with_ga():
    # example from pdf p. 6
    desired_sequence = np.array((0, 1, 4, 9, 16, 25, 36))
    # I think this one might not be possible with the current transformations
    # desired_sequence = np.array((9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
    # This should be possible
    # desired_sequence = np.array((4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
    base_sequence = np.ones(desired_sequence.shape, dtype=desired_sequence.dtype)
    objective = make_sequence_objective(desired_sequence, base_sequence)
    optimizer = ga.GAOptimizer(objective,
                               next_transformation_gen,  # type: ignore
                               [np.random.randint(len(NUM_TO_TRANSFORMATION), size=8)
                                for _ in range(100)],
                               True)

    population_with_fitness = []
    diversities = []
    costs = []
    pbar = tqdm(range(6000))
    global_best = None
    for _ in pbar:
        population_with_fitness = optimizer.step()
        unique_genotypes = {tuple(genotype) for genotype in map(lambda x: x[1], population_with_fitness)}
        diversities.append(len(unique_genotypes)/len(population_with_fitness))
        iteration_best = min(population_with_fitness, key=lambda x: x[0])
        costs.append(iteration_best[0])
        if global_best is None or iteration_best[0] < global_best[0]:
            global_best = iteration_best
        pbar.set_description('cost: {} div {:.2f}'.format(costs[-1], diversities[-1]))
        if global_best[0] == 0:
            break

    print('answer:', global_best[1]%len(NUM_TO_TRANSFORMATION))
    print('sequence:', trans_inds_to_seq(global_best[1]%len(NUM_TO_TRANSFORMATION), base_sequence))
    print('cost:', global_best[0])
    report_on_ga(costs, diversities)


def next_transformation_gen(transforms: Tuple[Tuple[int, np.ndarray], ...])\
        -> Tuple[np.ndarray, ...]:
    # couples = ga.roulette_wheel_rank_selection(transforms)
    couples = ga.roulette_wheel_cost_selection(transforms)
    # couples = ga.tournament_selection(transforms, 2)
    # couples = ga.uniform_random_pairing_selection(transforms)
    offspring = (ga.single_point_crossover(*couple) for couple in couples)
    children = tuple(child for pair in offspring for child in pair)
    mutate(children, .1)
    return children


def mutate(encodings: Tuple[np.ndarray, ...], prob: float):
    for i, j in it.product(range(len(encodings)), range(len(encodings[0]))):
        if np.random.rand() < prob:
            # encodings[i][j] += np.random.choice((-1, 1))
            encodings[i][j] = np.random.randint(len(NUM_TO_TRANSFORMATION))


def with_sa():
    desired_sequence = np.array((0, 1, 4, 9, 16, 25, 36))  # example from pdf p. 6
    # desired_sequence = np.array((9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
    base_sequence = np.ones(desired_sequence.shape, dtype=desired_sequence.dtype)
    transforms = np.zeros(6, dtype='int64')
    optimizer = sa.SAOptimizer(make_sequence_objective(desired_sequence, base_sequence),
                               sa.make_fast_schedule(1000),
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


def report_on_ga(costs, diversities) -> None:
    plt.title('Cost')
    plt.plot(costs)
    plt.figure()
    plt.title('Diversities')
    plt.plot(diversities)
    plt.show()


if __name__ == '__main__':
    main()
