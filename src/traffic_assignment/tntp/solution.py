from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, List, Tuple

import numpy as np

from . import common


@dataclass
class TNTPSolution:
    links: List[SolutionLink]

    @classmethod
    def read_text(cls, contents: str) -> TNTPSolution:
        lines = filter(is_data_line, contents.splitlines())
        links = sorted(map(SolutionLink.from_line, lines))
        return TNTPSolution(links)

    def link_flow(self) -> np.array:
        return np.array([l.volume for l in self.links])

    def link_cost(self) -> np.array:
        return np.array([l.cost for l in self.links])


class SolutionLink(NamedTuple):
    origin: int
    destination: int
    volume: float
    cost: float

    @classmethod
    def from_line(cls, line: str) -> SolutionLink:
        items = line.rstrip(common.END_OF_LINE).strip().split(common.DELIMITER)
        orgn, dest, volume, cost = items
        return SolutionLink(
            int(orgn),
            int(dest),
            float(volume),
            float(cost),
        )

    @property
    def id(self) -> Tuple[int, int]:
        return self.origin, self.destination


def is_data_line(line: str) -> bool:
    return not (
        common.is_empty(line)
        or common.is_metadata_line(line)
        or common.is_comment(line)
    ) and is_numeric(line)


def is_numeric(line: str) -> bool:
    try:
        float(line.strip().split(common.DELIMITER)[0])
    except ValueError:
        return False
    else:
        return True
