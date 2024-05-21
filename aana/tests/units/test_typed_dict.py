# ruff: noqa: S101

from typing import TypedDict

from aana.utils.typing import is_typed_dict


def test_is_typed_dict() -> None:
    """Test the is_typed_dict function."""

    class Foo(TypedDict):
        foo: str
        bar: int

    assert is_typed_dict(Foo) == True


def test_is_not_typed_dict():
    """Test the is_typed_dict function."""

    class Foo:
        pass

    assert is_typed_dict(Foo) == False
    assert is_typed_dict(dict) == False
