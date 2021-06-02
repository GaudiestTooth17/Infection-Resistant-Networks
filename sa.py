from typing import Callable, Sequence, Union, Tuple
import numpy as np
import time
import matplotlib.pyplot as plt


def main():
    sa_demo()


def make_sa_optimizer(objective: Callable, next_temp: Callable, neighbor: Callable,
                      sigma0: Union[Sequence[int], Sequence[float]]) -> Callable[[], Tuple[Sequence, float]]:
    T = next_temp()
    sigma = sigma0

    def step():
        nonlocal sigma, T
        sigma_prime = neighbor(sigma)
        energy = objective(sigma)
        energy_prime = objective(sigma_prime)
        curr_energy = energy
        if P(energy, energy_prime, T) >= np.random.rand():
            sigma = sigma_prime
            curr_energy = energy_prime
        T = next_temp()

        return sigma, curr_energy

    return step


def P(energy, energy_prime, T) -> float:
    acceptance_prob = 1.0 if energy_prime < energy else np.exp(-(energy_prime-energy)/T)  # type: ignore
    return acceptance_prob


def make_fast_schedule(T0: float) -> Callable:
    num_steps = -1

    def next_temp():
        nonlocal num_steps
        num_steps += 1
        return T0 / (num_steps + 1)

    return next_temp


def make_linear_schedule(T0: float, delta_T: float) -> Callable[[], float]:
    T = T0 + delta_T

    def schedule() -> float:
        nonlocal T
        T -= delta_T
        return max(0, T)

    return schedule


def example_objective(sigma: np.ndarray) -> int:
    size = sigma.shape[0]
    fitness = 10*size
    for i, x in enumerate(sigma):  # type: ignore
        fitness -= abs(i-x)
    return -fitness


def example_neighbor(solution: np.ndarray) -> np.ndarray:
    new = np.copy(solution)
    new[np.random.randint(0, new.shape[0])] += np.random.choice((-1, 1))
    return new


def sa_demo():
    start_time = time.time()
    print('Beginning.')
    sequence_length = 10
    T0 = 100.0
    max_steps = 500
    sigma0 = np.ones(sequence_length, dtype='int')
    optimizer_step = make_sa_optimizer(example_objective, make_fast_schedule(T0),
                                       example_neighbor, sigma0)

    best_solution = None
    energies = np.zeros(max_steps)
    for step in range(max_steps):
        best_solution, energy = optimizer_step()
        energies[step] = energy

    print(best_solution)
    print(f'Done. {time.time()-start_time}')
    plt.plot(energies)
    plt.show()


if __name__ == '__main__':
    main()
