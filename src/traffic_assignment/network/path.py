from typing import Iterable, Tuple
from marshmallow import Schema, fields, post_load
from functools import total_ordering

@total_ordering
class Path:
    def __init__(self, nodes: Iterable[int]):
        self.nodes = tuple(nodes)
        self.edges = tuple(self._generate_edges(nodes)) if self.nodes else []

    @staticmethod
    def _generate_edges(nodes) -> Iterable[Tuple[int, int]]:
        nodes = iter(nodes)
        u = next(nodes)
        for v in nodes:
            yield u, v
            u = v

    @property
    def origin(self):
        return self.nodes[0]

    @property
    def destination(self):
        return self.nodes[-1]

    def __repr__(self):
        return f"Path({self.nodes})"

    def __hash__(self):
        return hash(self.edges)

    def __eq__(self, other):
        return self.nodes == other.nodes

    def __lt__(self, other):
        return self.nodes < other.nodes


class PathSchema(Schema):
    nodes = fields.List(fields.Int)

    @post_load
    def make_path(self, data, **kwargs):
        return Path(data['nodes'])
