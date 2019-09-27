from __future__ import annotations

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from io import FileIO
from itertools import takewhile
from typing import List, Tuple
from typing import NamedTuple

METADATA_PATTERN = re.compile(r"<(.*)>\s+(.*)")
END_OF_LINE = ';'
ASSIGNMENT = ':'
DELIMITER = '\t'


def _tag(s: str) -> Tag:
    return Tag(f"<{s}>")


@dataclass(frozen=True)
class Tag:
    tag: str

    @property
    def key(self) -> str:
        return self.tag.strip('<>')


class MetaDataTags(NamedTuple):
    end: Tag = _tag('END OF METADATA')
    number_of_zones: Tag = _tag('NUMBER OF ZONES')
    number_of_nodes: Tag = _tag('NUMBER OF NODES')
    number_of_links: Tag = _tag('NUMBER OF LINKS')
    first_thru_node: Tag = _tag('FIRST THRU NODE')
    total_od_flow: Tag = _tag('TOTAL OD FLOW')


metadata_tags = MetaDataTags()


def is_metadata_line(line: str) -> bool:
    return line.startswith('<')


def is_end_of_metadata(line: str) -> bool:
    return line == metadata_tags.end.tag


def is_comment(line: str) -> bool:
    return line.startswith('~')


def is_empty(line: str) -> bool:
    return line.strip() == ''


def is_nonempty(line: str) -> bool:
    return not is_empty(line)


def is_header(line: str) -> bool:
    return is_metadata_line(line) or is_comment(line) or is_empty(line)


def parse_metadata(line: str) -> Tuple[str, str]:
    m = METADATA_PATTERN.match(line)
    key, value = m.groups()
    return key, value


def metadata(lines: List[str]) -> dict:
    metadata_lines = takewhile(
        lambda l: is_metadata_line(l) and not is_end_of_metadata(l),
        lines
    )
    return dict(map(parse_metadata, metadata_lines))


@dataclass
class TNTPDirectory:
    path: str

    def name(self) -> str:
        return os.path.basename(self.path)

    def path_of(self, ftype: TNTPFileType) -> str:
        return os.path.join(self.path, ftype.to_filename(self.name()))

    @contextmanager
    def _tntp_file(self, ftype: TNTPFileType) -> FileIO:
        with open(self.path_of(ftype)) as fp:
            yield fp

    @contextmanager
    def network_file(self) -> FileIO:
        with self._tntp_file(TNTPFileType.network) as fp:
            yield fp

    @contextmanager
    def solution_file(self) -> FileIO:
        with self._tntp_file(TNTPFileType.solution) as fp:
            yield fp

    @contextmanager
    def trips_file(self) -> FileIO:
        with self._tntp_file(TNTPFileType.trips) as fp:
            yield fp


class TNTPFileType(Enum):
    solution = 'flow'
    network = 'net'
    trips = 'trips'

    def to_filename(self, name: str) -> str:
        return f"{name}_{self.value}.tntp"
