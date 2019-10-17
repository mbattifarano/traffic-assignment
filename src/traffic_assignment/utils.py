import time
from typing import TypeVar, Optional, Union, MutableMapping, Iterator, Any
from dataclasses import dataclass
from marshmallow import Schema
import os
import gzip

import numpy as np
from io import FileIO

T = TypeVar('T')


def value_or_default(item: Optional[T], default: T) -> T:
    """Return item if it is not None, else return default."""
    if item is None:
        return default
    else:
        return item


class Timer:
    t0: float = None

    def start(self):
        self.t0 = time.time()
        return self

    def time_elapsed(self):
        return time.time() - self.t0


ArrayOrFloat = Union[np.ndarray, float]


def condense_array(a: np.ndarray) -> ArrayOrFloat:
    val = a[0]
    if (a == val).all():
        return val
    else:
        return a


@dataclass
class FileCache(MutableMapping):
    schema: Schema
    directory: str = 'cache'

    def __post_init__(self):
        try:
            os.mkdir(self.directory)
        except FileExistsError:
            # this is ok, directory already exists
            pass

    def _open(self, k: str, mode: str = 'r') -> FileIO:
        try:
            return open(self._file_of(k), mode)
        except FileNotFoundError:
            raise KeyError(f"No entry found for {k}.")

    def _file_of(self, k: str) -> str:
        return os.path.join(self.directory, k)

    def __setitem__(self, k: str, v: Any) -> None:
        with self._open(k, 'w') as fp:
            fp.write(self.schema.dumps(v))

    def __delitem__(self, k: str) -> None:
        os.remove(self._file_of(k))

    def __getitem__(self, k: str) -> Any:
        with self._open(k) as fp:
            return self.schema.loads(fp.read())

    def __len__(self) -> int:
        return len(os.listdir(self.directory))

    def __iter__(self) -> Iterator[Any]:
        return iter(os.listdir(self.directory))

    def clear(self):
        for k in list(self):
            del self[k]
