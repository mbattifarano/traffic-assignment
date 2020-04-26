from rtree.index import Index

SpatialIndex = Index


def insert_point(idx: SpatialIndex, id: int, x: float, y: float, obj=None):
    bbox = (x, y, x, y)
    return idx.insert(id, bbox, obj=obj)


def nearest_neighbor(idx: SpatialIndex, x: float, y: float):
    bbox = (x, y, x, y)
    items = idx.nearest(bbox, 1, objects=True)
    try:
        return next(items)
    except StopIteration:
        return None
