from __future__ import annotations
from typing import NamedTuple, Tuple

from .node import Node


class Link(NamedTuple):
    id: int
    origin: Node
    destination: Node

    @property
    def edge(self) -> Tuple[str, str]:
        return self.origin.name, self.destination.name
