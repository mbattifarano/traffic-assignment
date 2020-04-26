from .schemas import *
from .link import Link, MIN_CAPACITY, INF_CAPACITY
from .node import Node, Coordinate
from typing import Tuple, Mapping
import numpy as np


def capacity(cap: int) -> int:
    return np.clip(cap, MIN_CAPACITY, INF_CAPACITY)


def to_links(properties: Mapping, geometry: dict) -> Tuple[Link, Link]:
    a = Coordinate(
        latitude=properties[LinkProperties.latitude_a],
        longitude=properties[LinkProperties.longitude_a],
    )
    b = Coordinate(
        latitude=properties[LinkProperties.latitude_b],
        longitude=properties[LinkProperties.longitude_b]
    )
    ab_link = Link(
        id=int(properties[LinkProperties.id]),
        length=properties[LinkProperties.length],
        speed=properties[LinkProperties.speed_ab],
        capacity=capacity(properties[LinkProperties.capacity_ab]),
        zone=properties[LinkProperties.zone_ab],
        lanes=properties[LinkProperties.lanes_ab],
        from_point=a,
        to_point=b,
        geometry=geometry,
    )
    reverse_geom = geometry.copy()
    reverse_geom[Geometry.coordinates] = list(reversed(geometry[Geometry.coordinates]))
    ba_link = Link(
        id=int(properties[LinkProperties.id]),
        length=properties[LinkProperties.length],
        speed=properties[LinkProperties.speed_ba],
        capacity=capacity(properties[LinkProperties.capacity_ba]),
        zone=properties[LinkProperties.zone_ba],
        lanes=properties[LinkProperties.lanes_ba],
        from_point=b,
        to_point=a,
        geometry=reverse_geom,
    )
    return ab_link, ba_link


def to_node(id: int, properties: Mapping) -> Node:
    return Node(
        id=id,
        name=int(properties[NodeProperties.id]),
        latitude=properties[NodeProperties.latitude],
        longitude=properties[NodeProperties.longitude],
    )
