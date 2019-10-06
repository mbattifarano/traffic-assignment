from typing import Iterable, Tuple


class Path:
    def __init__(self, nodes: Iterable[str]):
        self.nodes = tuple(nodes)
        self.edges = tuple(self._generate_edges(nodes)) if self.nodes else []

    @staticmethod
    def _generate_edges(nodes) -> Iterable[Tuple[str, str]]:
        nodes = iter(nodes)
        u = next(nodes)
        for v in nodes:
            yield u, v
            u = v

    def __hash__(self):
        return hash(self.edges)
