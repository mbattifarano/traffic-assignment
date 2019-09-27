import time
from typing import TypeVar, Optional, Union
import numpy as np

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

    def time_elapsed(self):
        return time.time() - self.t0


ArrayOrFloat = Union[np.ndarray, float]


def condense_array(a: np.ndarray) -> ArrayOrFloat:
    val = a[0]
    if (a == val).all():
        return val
    else:
        return a
