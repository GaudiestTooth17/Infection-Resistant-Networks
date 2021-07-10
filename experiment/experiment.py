"""
This file is for running experiments and reporting on them.
"""
from experiment.connected_community import poisson_entry_point, uniform_entry_point
from experiment.agent_generated import agent_generated_entry_point
from experiment.social_circles import social_circles_entry_point


def main():
    # poisson_entry_point()
    # uniform_entry_point()
    # agent_generated_entry_point()
    social_circles_entry_point()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nGood bye.')
    except EOFError:
        print('\nGood bye.')
