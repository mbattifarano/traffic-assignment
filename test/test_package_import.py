import pytest


def test_package_is_importable():
    try:
        import traffic_assignment
    except ModuleNotFoundError:
        pytest.fail("`traffic_assignment` cannot be imported.")
