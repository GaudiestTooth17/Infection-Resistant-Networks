from customtypes import Number
from unittest import TestCase
from genfuncs import identity, make_scaler, make_right_shift, differentiation, summation


class TestGenfuncs(TestCase):
    def test_identity(self):
        sequence = (1, 1, 1, 1, 2)
        actual = tuple(identity(sequence))

        self.assertEqual(actual, sequence, f'Expected: {sequence}\nReceived: {actual}')
    
    def test_scale(self):
        sequence = (1, 2, 3, 4)
        for c in range(10):
            scale = make_scaler(c)
            expected = tuple(c*x for x in sequence)
            actual = tuple(scale(sequence))
            self.assertEqual(actual, expected, f'Failed with c == {c}.\nExpected: {expected}\nActual: {actual}')
    
    def test_right_shift(self):
        sequence = (1, 2, 0)

        shift1 = make_right_shift(1)
        expected1 = (0, 1, 2, 0)
        actual1 = tuple(shift1(sequence))
        self.assertEqual(actual1, expected1, basic_msg(actual1, expected1))

        shift3 = make_right_shift(3)
        expected3 = (0, 0, 0, 1, 2, 0)
        actual3 = tuple(shift3(sequence))
        self.assertEqual(actual3, expected3, basic_msg(actual3, expected3))

    def test_differentation(self):
        sequence = (16, 9, 4, 1)
        expected = (9, 8, 3)
        actual = tuple(differentiation(sequence))
        self.assertEqual(actual, expected, basic_msg(actual, expected))

    def test_summation(self):
        sequence0 = (1, 1, 1, 1)
        expected0 = (1, 2, 3, 4)
        actual0 = tuple(summation(sequence0))
        self.assertEqual(actual0, expected0, basic_msg(actual0, expected0))

        sequence1 = (5, 3, 6, 7)
        expected1 = (5, 8, 14, 21)
        actual1 = tuple(summation(sequence1))
        self.assertEqual(actual1, expected1, basic_msg(actual1, expected1))


def basic_msg(actual, expected) -> str:
    return f'Expected: {expected}\nActual: {actual}'
