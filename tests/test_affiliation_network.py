import sys
sys.path.append('')
from unittest import TestCase
from networkgen import make_affiliation_network
import itertools as it
import numpy as np


class TestAssociationNetwork(TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(66)
        self.N = 1000

    def test_one_group(self):
        """
        Test that 1 group with 100% membership will create a complete network
        """
        group_to_membership = [1.0]
        expected_edges = set(it.combinations(range(self.N), 2))
        net = make_affiliation_network(group_to_membership, self.N, self.rng)

        self.assertEqual(net.N, self.N, f'Expected {self.N} nodes, got {net.N} nodes')
        for edge in expected_edges:
            with self.subTest('Searching for edge', e=edge):
                self.assertTrue(net.G.has_edge(edge[0], edge[1]),
                                f'Network does not contain {edge}')

    def test_no_groups(self):
        """
        Test that having no groups will create a disconnected network
        """
        group_to_membership = []
        net = make_affiliation_network(group_to_membership, self.N, self.rng)
        edges = tuple(net.G.edges)
        self.assertEqual(net.N, self.N, f'Expected {self.N} nodes, got {net.N} nodes')
        self.assertTrue(len(edges) == 0, f'Expected no edges, got {len(edges)} edges')

    def test_right_number_of_edges(self):
        """
        """
        membership_perc = .25
        group_to_membership = [membership_perc]
        expected_edges = int((1/32)*(self.N**2) - (1/8)*self.N)
        net = make_affiliation_network(group_to_membership, self.N, self.rng)

        self.assertEqual(net.N, self.N, f'Expected {self.N} nodes, got {net.N} nodes')

        actual_edges = len(tuple(net.G.edges))
        leeway = expected_edges // 100  # Expected result can be 1% off in either direction
        self.assertTrue(expected_edges-leeway <= actual_edges <= expected_edges+leeway,
                        f'Expected about {expected_edges} +/- {leeway} edges, '
                        f'got {actual_edges} edges')
