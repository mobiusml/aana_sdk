# ruff: noqa: S101

from typing import TypedDict

from aana.utils.typing import is_typed_dict


def test_is_typed_dict():
    """Test the is_typed_dict function."""
    class Foo(TypedDict):
        foo: str
        bar: int

    s:  Foo = { 'foo': 'foo', 'bar': 0 }

    assert is_typed_dict(s) == True

def test_is_not_typed_dict():
    """Test the is_typed_dict function."""
    s = {'foo': 'foo', 'bar': 0}
    assert is_typed_dict(s) == False