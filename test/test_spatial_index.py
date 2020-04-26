from hypothesis import given, infer
from traffic_assignment.spatial_index import *


@given(x=infer, y=infer)
def test_insert_point(x: int, y: int):
    idx = SpatialIndex()
    insert_point(idx, 0, x, y)
    bbox = idx.bounds
    assert idx.count(bbox) == 1


@given(x=infer, y=infer)
def test_nearest_neighbor(x: int, y: int):
    idx = SpatialIndex()
    insert_point(idx, 0, x, y, 'foo')
    insert_point(idx, 1, x+1, y, 'bar')
    item = nearest_neighbor(idx, x, y)
    assert item.id == 0
    assert item.object == 'foo'
