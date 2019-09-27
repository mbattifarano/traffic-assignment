from hypothesis import given
from hypothesis.strategies import text, tuples, lists, integers, characters

from traffic_assignment.tntp import common


tokens = text(characters(blacklist_characters='\n'), min_size=1)


@given(tokens, tokens)
def test_parse_metadata_line(key, value):
    line = f"<{key}> {value}"
    assert common.is_metadata_line(line)
    assert common.parse_metadata(line) == (key, value.lstrip())


@given(lists(tuples(tokens, tokens)), integers(min_value=0, max_value=2**16))
def test_metadata(lines, end_index):
    end_tag = "<END OF METADATA>"
    lines = [f"<{k}> {v}" for k, v in lines]
    lines.insert(end_index, end_tag)
    parsed = common.metadata(lines)
    assert len(parsed) <= lines.index(end_tag)
