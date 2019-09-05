from typing import TypeVar, Optional

T = TypeVar('T')


def value_or_default(item: Optional[T], default: T) -> T:
    """Return item if it is not None, else return default."""
    if item is None:
        return default
    else:
        return item
