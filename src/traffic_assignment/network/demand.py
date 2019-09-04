from __future__ import annotations
from typing import NamedTuple

from .node import Node


class Demand(NamedTuple):
    origin: Node
    destination: Node
    volume: float
