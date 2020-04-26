from typing import NamedTuple


class Coordinate(NamedTuple):
    latitude: float
    longitude: float


class Node(NamedTuple):
    id: int
    name: int
    latitude: float
    longitude: float

    @property
    def coordinate(self) -> Coordinate:
        return Coordinate(self.latitude, self.longitude)
