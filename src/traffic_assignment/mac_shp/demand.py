import numpy as np
from typing import NamedTuple, Iterable
from itertools import product


class Demand(NamedTuple):
    from_zone: int
    to_zone: int
    volume: float


def read_csv(filename: str) -> np.ndarray:
    return np.genfromtxt(
        filename,
        delimiter=',',
    )


def to_zone(i: int) -> int:
    return i + 1


def nonzeros(a: np.ndarray):
    return zip(*a.nonzero())


def to_demand(od_matrix: np.ndarray) -> Iterable[Demand]:
    for i, j in nonzeros(od_matrix):
        volume = od_matrix[i, j]
        yield Demand(
            to_zone(i),
            to_zone(j),
            volume
        )
