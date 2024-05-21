# ruff: noqa: S101

import pytest
from pydantic import BaseModel

from aana.core.models.base import merged_options


class MyOptions(BaseModel):
    """Test option class."""

    field1: str
    field2: int | None = None
    field3: bool


def test_merged_options_same_type():
    """Test merged_options with options of the same type as default_options."""
    default = MyOptions(field1="default1", field2=2, field3=True)
    to_merge = MyOptions(field1="merge1", field2=None, field3=False)
    merged = merged_options(default, to_merge)

    assert merged.field1 == "merge1"
    assert (
        merged.field2 == 2
    )  # Should retain value from default_options as it's None in options
    assert merged.field3 == False


def test_merged_options_none():
    """Test merged_options with options=None."""
    default = MyOptions(field1="default1", field2=2, field3=True)
    merged = merged_options(default, None)

    assert merged.model_dump() == default.model_dump()


def test_merged_options_type_mismatch():
    """Test merged_options with options of a different type from default_options."""

    class AnotherOptions(BaseModel):
        another_field: str

    default = MyOptions(field1="default1", field2=2, field3=True)
    to_merge = AnotherOptions(another_field="test")

    with pytest.raises(ValueError):
        merged_options(default, to_merge)
