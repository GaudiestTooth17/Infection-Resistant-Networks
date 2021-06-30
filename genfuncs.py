from customtypes import Number
from typing import Iterable, Callable
import itertools as it

Transformation = Callable[[Iterable[Number]], Iterable[Number]]


def identity(sequence: Iterable[Number]) -> Iterable[Number]:
    return sequence


def make_scaler(c: Number) -> Transformation:
    def scale(sequence: Iterable[Number]):
        return (x*c for x in sequence)
    return scale


def make_right_shift(distance: int) -> Transformation:
    def right_shift(sequence: Iterable[Number]):
        return it.chain((0 for _ in range(distance)), sequence)
    return right_shift


def differentiation(sequence: Iterable[Number]) -> Iterable[Number]:
    dropped_and_indexed = it.chain(it.dropwhile(lambda ix: ix[0] == 0, enumerate(sequence)))
    return (i*x for i, x in dropped_and_indexed)


def summation(sequence: Iterable[Number]) -> Iterable[Number]:
    return it.accumulate(sequence)
