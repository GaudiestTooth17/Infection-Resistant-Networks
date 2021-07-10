#!/usr/bin/python3
from networkgen._clique_gate import cgg_entry_point
from networkgen._social_circles import social_circles_entry_point
from networkgen._connected_community import connected_community_entry_point
from networkgen._agent_based import agent_based_entry_point


def main():
    # cgg_entry_point()
    # social_circles_entry_point()
    connected_community_entry_point()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nGood bye.')
    except EOFError:
        print('\nGood bye.')
