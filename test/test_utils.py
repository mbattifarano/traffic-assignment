from typing import Optional
import time
from hypothesis import given, infer
from traffic_assignment import utils
from marshmallow import Schema, fields
import pytest


@given(a=infer, b=infer)
def test_value_or_default(a: Optional[int], b: int):
    actual = utils.value_or_default(a, b)
    if a is None:
        assert actual == b
    else:
        assert actual == a


def test_timer():
    eps = 1e-2
    timer = utils.Timer()
    assert timer.t0 is None
    timer.start()
    time.sleep(0.5)
    assert abs(timer.time_elapsed() - 0.5) <= eps
    time.sleep(0.5)
    assert abs(timer.time_elapsed() - 1.0) <= eps
    timer.start()
    time.sleep(0.5)
    assert abs(timer.time_elapsed() - 0.5) <= eps


def test_file_cache():
    foo = dict(a=1, b=2)
    bar = dict(a=5, b=3)

    class FooSchema(Schema):
        a = fields.Integer()
        b = fields.Integer()

    cache = utils.FileCache(FooSchema(), 'test/artifacts/test_file_cache')
    cache.clear()
    assert len(cache) == 0
    with pytest.raises(KeyError):
        cache['foo']

    cache['foo'] = foo
    assert len(cache) == 1
    assert list(cache) == ['foo']
    assert cache['foo'] == foo

    cache['bar'] = bar
    assert len(cache) == 2
    assert list(cache) == ['foo', 'bar']
    assert cache['foo'] == foo
    assert cache['bar'] == bar

    del cache['foo']
    assert len(cache) == 1
    assert list(cache) == ['bar']
    assert cache['bar'] == bar
